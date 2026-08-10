// fllama_inference_queue.h — manages server_context instances and cancellation
#ifndef FLLAMA_INFERENCE_QUEUE_H
#define FLLAMA_INFERENCE_QUEUE_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "fllama.h"
#include <vector>

// Pulled in for the load-time tuning fields below (ggml_type,
// common_speculative_type), which are part of the server cache key.
#include "llama.cpp/common/common.h"
#include "llama.cpp/ggml/include/ggml.h"

// Forward declarations.
struct server_context;

// ---------------------------------------------------------------------------
// ServerResources — holds a server_context and its dedicated loop thread.
// ---------------------------------------------------------------------------
struct ServerResources {
  std::unique_ptr<server_context> srv_ctx;
  std::thread loop_thread;
  std::string model_path;
  std::chrono::steady_clock::time_point last_used;
  std::atomic<int> active_users{0};
  std::atomic<bool> shutting_down{false};

  // Params that require a fresh server_context if changed.
  int n_ctx         = 0;
  int n_gpu_layers  = -1;
  std::string mmproj_path;
  std::string draft_path; // MTP/speculative drafter model path ("" if none)
  int draft_n_max = 0;    // MTP/speculative max draft tokens; load-time param
  float draft_p_min = -1; // MTP/speculative min draft confidence; load-time param
  std::string lora_path;  // LoRA adapter path ("" if none); load-time param
  float lora_scale = 0;   // LoRA adapter strength; load-time param

  // Tuning overrides baked in when the model loads (runtime_overrides_json in
  // fllama.h): quantized KV cache, host-RAM prompt cache, n-gram
  // self-speculation. A server built with different values can't serve this
  // request, so they belong in the cache key.
  ggml_type cache_type_k = GGML_TYPE_F16;
  ggml_type cache_type_v = GGML_TYPE_F16;
  int cache_ram_mib = 0;
  int n_batch = 0;
  int n_ubatch = 0;
  std::vector<enum common_speculative_type> spec_types = {
      COMMON_SPECULATIVE_TYPE_NONE};

  ServerResources() = default;
  ~ServerResources(); // terminates loop, joins thread

  ServerResources(const ServerResources &) = delete;
  ServerResources &operator=(const ServerResources &) = delete;
};

// ---------------------------------------------------------------------------
// ServerManager — owns cached server_contexts, tracks cancellation.
//   No worker thread — callers run their own reader threads.
// ---------------------------------------------------------------------------
class ServerManager {
public:
  // Local fork patch: upstream defaults to 4 parallel server slots, which
  // splits the requested context window 4 ways and reserves 4× the KV cache
  // up front. This app is single-user — only one conversation generates at a
  // time — so the extra 3 slots are pure wasted RAM. On a 6GB iPhone a 3B
  // model at 4 slots reserved ~900MiB of KV and pushed the device into memory
  // pressure (visible stutter / freeze). 1 slot gives the conversation the
  // full requested window at a quarter of the KV memory. Post-answer title +
  // follow-up generations simply queue behind the answer instead of running
  // concurrently, which is imperceptible for those tiny (<100 token) calls.
  static constexpr int DEFAULT_N_PARALLEL = 1;
  static int MODEL_INACTIVITY_TIMEOUT_SEC;
  static int CLEANUP_INTERVAL_SEC;

  ServerManager();
  ~ServerManager();

  // Get or create a server_context for a model.  Increments active_users.
  // Returns nullptr on failure.  Thread-safe.
  ServerResources *get_or_create(const std::string &model_path,
                                 const common_params &params,
                                 fllama_log_callback logger);

  // Decrement active_users (call when your request finishes).
  void release(const std::string &model_path);

  // Mark a context as unsafe to reuse.  The active request still owns it until
  // release(); once active_users reaches zero the context is destroyed so the
  // next request recreates the backend from scratch.
  void mark_unhealthy(const std::string &model_path);

  // Destroy every idle server (active_users == 0) except [except_path]
  // (pass "" to evict all idle).  Frees model weights + KV immediately so a
  // different model can load without both being resident — on 6GB iPhones two
  // resident models plus a context re-create exceed the Metal working set and
  // the OS kills the app.  Thread-safe; destructors run outside the lock.
  void evict_idle_except(const std::string &except_path);

  // Cancellation.
  void cancel(int request_id);
  bool is_cancelled(int request_id);
  void clear_cancel(int request_id);

  // Track a live request thread so we can join on shutdown.
  void register_request_thread(int request_id, std::thread &&t);
  void unregister_request_thread(int request_id);

private:
  void cleanup_loop();

  mutable std::shared_mutex servers_lock;
  std::unordered_map<std::string, std::unique_ptr<ServerResources>> servers;

  std::mutex cancel_lock;
  std::unordered_set<int> cancelled;

  // Serialises all load_model() calls — ggml Metal init has global state
  // that is not safe for concurrent model loads.
  std::mutex model_load_mutex;

  std::mutex threads_lock;
  std::unordered_map<int, std::thread> request_threads;
  // Threads that finished but need joining.
  std::vector<std::thread> finished_threads;

  std::thread cleanup_thread;
  std::condition_variable_any cleanup_cond;
  std::atomic<bool> done{false};
};

#endif // FLLAMA_INFERENCE_QUEUE_H
