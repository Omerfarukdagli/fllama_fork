#ifndef FLLAMA_H
#define FLLAMA_H

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#if _WIN32
#define FFI_PLUGIN_EXPORT __declspec(dllexport)
#else
#define FFI_PLUGIN_EXPORT
#endif

#include <stdint.h> // For uint8_t

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*fllama_inference_callback)(const char *response, const char * openai_response_json_string, uint8_t done);
typedef void (*fllama_log_callback)(const char *);

struct fllama_gpu_memory_info {
  int32_t device_index;
  uint64_t total_bytes;
  uint64_t free_bytes;
  char name[128];
  char description[256];
  char device_id[128];
};

struct fllama_inference_request {
  int request_id; // Required: unique ID for the request. Used for cancellation.
  int context_size;        // Required: context size
  char *input;             // Required: input text
  int max_tokens;          // Required: max tokens to generate
  char *model_path;        // Required: .ggml model file path
  char *model_mmproj_path; // Optional: .mmproj file for multimodal models.
  int num_gpu_layers; // Required: number of GPU layers. 0 for CPU only. 99 for
                      // all layers. Automatically 0 on iOS simulator.
  int num_threads; // Required: 2 recommended. Platforms can be highly sensitive
                   // to this, ex. Android stopped working with 4 suddenly.
  float
      temperature; // Optional: temperature. Defaults to 0. (llama.cpp behavior)
  float top_p; // Optional: 0 < top_p <= 1. Defaults to 1. (llama.cpp behavior)
  float penalty_freq;   // Optional: 0 <= penalty_freq <= 1. Defaults to 0.0,
                        // which means disabled. (llama.cpp behavior)
  float penalty_repeat; // Optional: 0 <= penalty_repeat <= 1. Defaults to 1.0,
                        // which means disabled. (llama.cpp behavior)
  char *
      grammar; // Optional: BNF-like grammar to constrain sampling. Defaults to
               // "" (llama.cpp behavior). See
               // https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
  char *eos_token; // Optional: end of sequence token. Defaults to one in model file. (llama.cpp behavior)
                   // For example, in ChatML / OpenAI, <|im_end|> means the message is complete.
                   // Often times GGUF files were created incorrectly, and this should be overridden.
                   // Using fllamaChat from Dart handles this automatically.
  fllama_log_callback
      dart_logger; // Optional: Dart caller logger. Defaults to NULL.
  char * openai_request_json_string; // Optional: OpenAI JSON string. Defaults to NULL.
  char * draft_model_path; // Optional: MTP assistant/drafter GGUF for speculative
                           // decoding (e.g. gemma-4-*-it-assistant). NULL/"" disables.
                           // NOTE: keep draft KV cache at F16 (default); Q8 KV
                           // destroys MTP draft acceptance.
  int draft_n_max;         // Optional: tokens to draft per step when
                           // draft_model_path is set. <= 0 falls back to 3.
  float draft_p_min;       // Optional: minimum drafter top-token probability.
                           // < 0 uses llama.cpp default.
  char *lora_path;         // Optional: LoRA adapter GGUF loaded alongside the
                           // base model (personal fine-tunes). NULL/"" = none.
                           // The server keyed cache treats a different adapter
                           // as a different server, so switching reloads.
  float lora_scale;        // Optional: adapter strength. <= 0 uses 1.0.

  // Optional: JSON object of PER-REQUEST overrides. NULL/"" = none, which
  // leaves every value at llama.cpp's default — so a zero-initialized request
  // behaves exactly as it did before this field existed. Passing knobs as JSON
  // instead of individual struct fields keeps the ABI (and the ffigen'd Dart
  // struct) stable as we add tuning parameters.
  //
  // Recognized keys (any subset):
  //   "top_k"                        int    (llama.cpp default 40; <=0 = off)
  //   "min_p"                        float  (default 0.05; 0 = off)
  //   "typ_p" / "top_n_sigma"        float
  //   "penalty_last_n"               int    (default 64; 0 = off, -1 = n_ctx)
  //   "penalty_present"              float
  //   "dry_multiplier" / "dry_base"  float
  //   "dry_allowed_length"           int
  //   "seed"                         int    (fixed seed = reproducible runs;
  //                                          omitted = random, as before)
  //   "n_cache_reuse"                int    (min chunk size reused from the KV
  //                                          cache via shifting; 0 = off)
  //   "reasoning_budget_tokens"      int    (-1 = unlimited thinking)
  //   "reasoning_budget_start_tag"   string (e.g. "<think>")
  //   "reasoning_budget_end_tag"     string (e.g. "</think>")
  //   "reasoning_budget_message"     string (injected before the end tag when
  //                                          the budget runs out)
  char *request_overrides_json;

  // Optional: JSON object of LOAD-TIME overrides — these change how the model
  // context itself is built, so the server cache treats a different value as a
  // different server (same rule as lora_path). NULL/"" = none.
  //
  // Recognized keys (any subset):
  //   "cache_type_k" / "cache_type_v"  string  "f16" (default), "bf16",
  //                                            "q8_0", "q5_1", "q5_0",
  //                                            "q4_1", "q4_0", "iq4_nl"
  //   "cache_ram_mib"                  int     host-RAM prompt cache, MiB
  //                                            (0 = disabled, our default)
  //   "spec_ngram"                     string  self-speculative decoding with
  //                                            NO draft model: "simple",
  //                                            "map_k", "map_k4v", "mod",
  //                                            "cache". Omitted = off.
  //   "n_batch" / "n_ubatch"           int     prompt-processing batch sizes;
  //                                            they trade memory for
  //                                            time-to-first-token
  //
  // Unrecognized keys in either object are logged as warnings rather than
  // silently dropped — a knob nobody reads turns an experiment into a no-op
  // that still looks like a result.
  char *runtime_overrides_json;
};

EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT void fllama_inference(struct fllama_inference_request request,
                                        fllama_inference_callback callback);
EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT void fllama_inference_sync(struct fllama_inference_request request,
                           fllama_inference_callback callback);
EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT void fllama_inference_cancel(int request_id);

// ── Embeddings and reranking ────────────────────────────────────────────────
//
// llama.cpp has carried both for a long time (llama_set_embeddings,
// llama_get_embeddings_seq, LLAMA_POOLING_TYPE_RANK) and the vendored server
// already implements SERVER_TASK_TYPE_EMBEDDING / _RERANK. fllama simply never
// exposed them, so callers had to fall back to lexical search. This is the same
// shape as request_overrides_json: the capability was already in the engine.
//
// A context can only do ONE of these: llama.cpp decides at load time whether it
// pools (embeddings) or generates (chat). The server cache therefore keys on
// this flag, exactly like a LoRA adapter — an embedding context will never be
// handed to a chat request and vice versa. In practice the embedding model is a
// separate, small GGUF anyway.
struct fllama_embed_request {
  int request_id;    // Required: unique ID (used for cancellation).
  int context_size;  // Required: context size.
  char *model_path;  // Required: embedding/reranking .gguf.
  int num_gpu_layers;
  int num_threads;

  // Required: JSON describing the work. Two shapes, picked by which keys are
  // present, so one entry point serves both:
  //
  //   Embedding:
  //     {"input": ["passage one", "passage two"],
  //      "pooling": "mean",      // "mean" (default) | "cls" | "last"
  //      "normalize": true}      // L2-normalize; default true, which is what
  //                              // cosine similarity expects
  //
  //   Reranking:
  //     {"query": "warranty period",
  //      "documents": ["...", "..."]}
  //     Pooling is forced to "rank" — a reranker attaches a classification
  //     head, so any other pooling would silently return nonsense.
  char *input_json;

  fllama_log_callback dart_logger; // Optional.
};

// Called once when the work finishes.
// [result_json] on success:
//   embedding: {"embeddings": [[...floats...], ...], "n_tokens": 123}
//   rerank:    {"scores": [0.81, 0.12], "n_tokens": 123}
// [error] is NULL on success, and a message when result_json is NULL.
typedef void (*fllama_embed_callback)(const char *result_json,
                                      const char *error);

EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT void
fllama_embed(struct fllama_embed_request request,
             fllama_embed_callback callback);

// Frees every idle (no in-flight request) model context except the one at
// [except_model_path] (pass NULL or "" to evict all idle).  Host apps call
// this before loading a DIFFERENT model (model switch / benchmark) so two
// models are never resident at once — on memory-tight phones two resident
// models plus a context re-create exceed the GPU working set and the OS
// kills the app.
EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT void
fllama_evict_idle_servers(const char *except_model_path);

// GPU device information.
// Returns the number of GPU devices visible to ggml/llama.cpp.
EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT int fllama_get_gpu_device_count(void);

// Fills [out_info] for the GPU at [gpu_index].
// Returns 0 on success, non-zero on failure.
EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT int fllama_get_gpu_memory_info(
    int gpu_index,
    struct fllama_gpu_memory_info * out_info);
#ifdef __cplusplus
}
#endif

#endif // FLLAMA_H