// fllama.cpp — Phase 3: thin adapter over llama.cpp's server_context
//
// fllama_inference() spawns a reader thread that posts a server_task and
// streams results back via the Dart callback.  Multiple concurrent calls
// are batched automatically by server_context::update_slots().

#include "fllama.h"
#include "fllama_inference_queue.h"
#include "fllama_mtmd.h"

// server-context headers (no HTTP / httplib dependency)
#include "server-context.h"
#include "server-task.h"
#include "server-common.h"

#include "llama.cpp/common/chat.h"
#include "llama.cpp/common/common.h"
#include "llama.cpp/ggml/include/ggml.h"
#include "llama.cpp/include/llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "ggml-backend.h"

// ── Tuning overrides (request_overrides_json / runtime_overrides_json) ───────
//
// Both are optional JSON objects; an absent key leaves llama.cpp's own default
// untouched, so an empty/NULL string reproduces the pre-override behaviour
// exactly. See fllama.h for the recognized keys.

using fllama_json = nlohmann::ordered_json;

// Defined in the logging section just below.
static void log_message(const char *msg, fllama_log_callback logger);
static void log_message(const std::string &m, fllama_log_callback l);

static fllama_json parse_overrides(const char *s, const char *what,
                                   fllama_log_callback logger) {
  if (!s || s[0] == '\0') return fllama_json::object();
  try {
    auto j = fllama_json::parse(s);
    if (!j.is_object()) throw std::runtime_error("not a JSON object");
    return j;
  } catch (const std::exception &e) {
    // A malformed override must never take the request down with it — run with
    // defaults and say so in the log.
    log_message(std::string("[fllama] ignoring bad ") + what + ": " + e.what(),
                logger);
    return fllama_json::object();
  }
}

// Applies one key and records that it was recognized. The bookkeeping exists
// because the dangerous failure here is silent: a key nobody reads makes the
// caller's experiment a no-op, and "no difference" then gets written down as a
// real result. Anything left unconsumed is logged loudly instead.
template <typename T>
static void apply_override(const fllama_json &j, const char *key, T &out,
                           std::vector<std::string> *seen = nullptr) {
  if (seen) seen->push_back(key);
  auto it = j.find(key);
  if (it != j.end() && !it->is_null()) out = it->get<T>();
}

static void warn_unknown_keys(const fllama_json &j,
                              const std::vector<std::string> &known,
                              const char *what, fllama_log_callback logger) {
  for (auto it = j.begin(); it != j.end(); ++it) {
    if (std::find(known.begin(), known.end(), it.key()) == known.end()) {
      log_message(std::string("[fllama] WARNING: ") + what +
                      " key ignored (not implemented): " + it.key(),
                  logger);
    }
  }
}

// Mirrors the (file-static, so unreachable) kv_cache_type_from_str in
// common/arg.cpp. Unknown names keep the caller's current value.
static bool kv_cache_type_from_name(const std::string &s, ggml_type &out) {
  static const std::pair<const char *, ggml_type> kTypes[] = {
      {"f32", GGML_TYPE_F32},   {"f16", GGML_TYPE_F16},
      {"bf16", GGML_TYPE_BF16}, {"q8_0", GGML_TYPE_Q8_0},
      {"q4_0", GGML_TYPE_Q4_0}, {"q4_1", GGML_TYPE_Q4_1},
      {"iq4_nl", GGML_TYPE_IQ4_NL}, {"q5_0", GGML_TYPE_Q5_0},
      {"q5_1", GGML_TYPE_Q5_1},
  };
  for (const auto &t : kTypes) {
    if (s == t.first) { out = t.second; return true; }
  }
  return false;
}

static bool spec_ngram_type_from_name(const std::string &s,
                                      common_speculative_type &out) {
  // Self-speculative (n-gram) decoding needs NO second model — it drafts from
  // patterns already present in the context, so it costs no extra weights on a
  // memory-bound phone.
  static const std::pair<const char *, common_speculative_type> kTypes[] = {
      {"simple", COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
      {"map_k", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
      {"map_k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
      {"mod", COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
      {"cache", COMMON_SPECULATIVE_TYPE_NGRAM_CACHE},
  };
  for (const auto &t : kTypes) {
    if (s == t.first) { out = t.second; return true; }
  }
  return false;
}

// Load-time overrides: these change how the llama_context is built, so the
// server cache must treat a different value as a different server.
static void apply_runtime_overrides(const fllama_json &j, common_params &params,
                                    fllama_log_callback logger) {
  std::string k_type, v_type, ngram;
  std::vector<std::string> known;
  apply_override(j, "cache_type_k", k_type, &known);
  apply_override(j, "cache_type_v", v_type, &known);
  apply_override(j, "cache_ram_mib", params.cache_ram_mib, &known);
  apply_override(j, "spec_ngram", ngram, &known);
  apply_override(j, "n_batch", params.n_batch, &known);
  apply_override(j, "n_ubatch", params.n_ubatch, &known);
  warn_unknown_keys(j, known, "runtime_overrides_json", logger);

  if (!k_type.empty() && !kv_cache_type_from_name(k_type, params.cache_type_k))
    log_message("[fllama] unknown cache_type_k: " + k_type, logger);
  if (!v_type.empty() && !kv_cache_type_from_name(v_type, params.cache_type_v))
    log_message("[fllama] unknown cache_type_v: " + v_type, logger);

  if (!ngram.empty()) {
    common_speculative_type t;
    if (spec_ngram_type_from_name(ngram, t)) {
      params.speculative.types = {t};
    } else {
      log_message("[fllama] unknown spec_ngram: " + ngram, logger);
    }
  }
}

// ── Logging ──────────────────────────────────────────────────────────────────

static constexpr bool kLogVerboseDebugMessages = false;

static void log_message(const char *msg,
                        fllama_log_callback logger = nullptr) {
  if (!logger) {
    fprintf(stderr, "%s\n", msg);
    fflush(stderr);
    return;
  }
  static std::mutex mtx;
  static std::deque<std::string> q;
  std::string s(msg);
  for (size_t p = 0; (p = s.find('\n', p)) != std::string::npos; p += 4)
    s.replace(p, 1, "[NL]");
  std::lock_guard<std::mutex> lk(mtx);
  q.push_back(std::move(s));
  while (q.size() > 1000)
    q.pop_front();
  logger(q.back().c_str());
}
static void log_message(const std::string &m,
                        fllama_log_callback l = nullptr) {
  log_message(m.c_str(), l);
}

static bool fllama_is_noisy_per_token_llama_log(const char *text) {
  if (!text) {
    return false;
  }
  std::string s(text);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
    s.pop_back();
  }
  return s == "set_embeddings: value = 0" ||
         s == "set_adapters_lora: adapters = 0" ||
         s == "adapters_lora_are_same: adapters = 0";
}

// llama.cpp's own errors go to stderr, which never reaches Dart — so a failed
// load surfaced as a bare "Failed to create inference context" with no reason,
// for us AND for the user. Keep the last error line so it can be attached to
// the message we send back.
static std::mutex g_last_error_mutex;
static std::string g_last_llama_error;

static void fllama_note_llama_error(const char *text) {
  if (text == nullptr) {
    return;
  }
  std::string line(text);
  while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
    line.pop_back();
  }
  if (line.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_last_error_mutex);
  g_last_llama_error = line;
}

std::string fllama_take_last_llama_error() {
  std::lock_guard<std::mutex> lock(g_last_error_mutex);
  std::string out = g_last_llama_error;
  g_last_llama_error.clear();
  return out;
}

static void fllama_filtered_llama_log_callback(enum ggml_log_level level,
                                               const char *text,
                                               void *user_data) {
  if (fllama_is_noisy_per_token_llama_log(text)) {
    return;
  }
  if (level == GGML_LOG_LEVEL_ERROR) {
    fllama_note_llama_error(text);
  }
  common_log_default_callback(level, text, user_data);
}

struct fllama_callback_payload {
  std::string response;
  std::string openai_json;
};

static void emit_inference_callback(fllama_inference_callback callback,
                                    std::string response,
                                    std::string openai_json,
                                    uint8_t done) {
  if (!callback) {
    return;
  }

  // NativeCallable.listener relays calls from worker threads back to the Dart
  // isolate asynchronously. Pointer arguments are just addresses, so c_str()
  // into a local std::string can be dead by the time Dart reads it. Keep a
  // rolling native-side copy alive, same as log_message() does for logger
  // callbacks above.
  static std::mutex mtx;
  static std::deque<fllama_callback_payload> q;
  std::lock_guard<std::mutex> lk(mtx);
  q.push_back({std::move(response), std::move(openai_json)});
  while (q.size() > 1000) {
    q.pop_front();
  }
  const auto &payload = q.back();
  callback(payload.response.c_str(), payload.openai_json.c_str(), done);
}

// ── Globals ──────────────────────────────────────────────────────────────────

// Intentionally leaked — avoids static destruction order crash on exit.
// (ggml Metal statics may be destroyed before g_mgr's destructor runs,
//  causing ggml_abort when server_context tries to free Metal resources.)
static ServerManager &g_mgr = *new ServerManager();
static std::once_flag  g_backend_init;

static void fllama_backend_init_once() {
  std::call_once(g_backend_init, [] {
    // Filter llama.cpp's default logger: without a callback it prints EVERY
    // level to stderr, including per-token LLAMA_LOG_DEBUG lines
    // (set_embeddings / set_adapters_lora / ...) — thousands of lines per
    // generation that flood the host app's console and visibly jank debug
    // builds. Keep WARN+ (and CONT lines that continue a kept message).
    llama_log_set(
        [](enum ggml_log_level level, const char *text, void * /*ud*/) {
          static thread_local bool last_kept = false;
          if (level == GGML_LOG_LEVEL_CONT) {
            if (last_kept) fputs(text, stderr);
            return;
          }
          last_kept = (level >= GGML_LOG_LEVEL_WARN);
          if (last_kept) fputs(text, stderr);
        },
        nullptr);
    ggml_backend_load_all();
    llama_backend_init();
    llama_log_set(fllama_filtered_llama_log_callback, nullptr);
  });
}

static std::vector<ggml_backend_dev_t> fllama_get_gpu_devices() {
  fllama_backend_init_once();

  std::vector<ggml_backend_dev_t> devices;
  for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
    auto * dev = ggml_backend_dev_get(i);
    if (dev == nullptr) {
      continue;
    }
    if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
      continue;
    }
    devices.push_back(dev);
  }
  return devices;
}

static void fllama_copy_cstr(char * dst, size_t cap, const char * src) {
  if (dst == nullptr || cap == 0) {
    return;
  }
  std::snprintf(dst, cap, "%s", src ? src : "");
}

static bool fllama_error_requires_backend_recreation(const std::string &msg) {
  // llama.cpp reports Metal command-buffer OOM / poisoned backend failures to
  // callers as a generic "Compute error.".  Once the backend is in that state,
  // the fix is to destroy and recreate the server_context rather than retrying
  // on the same context.
  return msg.find("Compute error") != std::string::npos ||
         msg.find("backend is in error state") != std::string::npos ||
         msg.find("failed to compute graph") != std::string::npos ||
         msg.find("failed to decode") != std::string::npos ||
         msg.find("OutOfMemory") != std::string::npos ||
         msg.find("out of memory") != std::string::npos;
}

// ── The actual inference logic (runs on per-request thread) ──────────────────

static void run_inference(fllama_inference_request request,
                          fllama_inference_callback callback) {
  try {
    int64_t t0 = ggml_time_ms();
    log_message("[fllama] Inference start", request.dart_logger);

    // One-time backend init.
    fllama_backend_init_once();

    // ── 1. Build common_params ────────────────────────────────────────

    common_params params;
    params.model.path       = request.model_path;
    params.n_ctx            = request.context_size;
    // Match llama.cpp server defaults more closely instead of tying batch
    // sizes to the full context window.
    params.n_batch          = std::min<int32_t>(request.context_size, 2048);
    params.n_ubatch         = std::min<int32_t>(params.n_batch, 512);
    params.flash_attn_type  = LLAMA_FLASH_ATTN_TYPE_AUTO;
    params.n_parallel       = ServerManager::DEFAULT_N_PARALLEL;
    params.n_predict        = request.max_tokens;
    params.sampling.temp    = request.temperature;
    params.sampling.top_p   = request.top_p;
    params.sampling.penalty_freq   = request.penalty_freq;
    params.sampling.penalty_repeat = request.penalty_repeat;
    params.cpuparams.n_threads     = request.num_threads;
    params.use_jinja = true;
    params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;

    // Personal LoRA adapter, loaded next to the base model. The server applies
    // it at load time, so a request that names one must not reuse a server
    // that was created without it — see the cache key in
    // fllama_inference_queue.cpp.
    if (request.lora_path != nullptr && request.lora_path[0] != '\0') {
      common_adapter_lora_info lora;
      lora.path  = request.lora_path;
      lora.scale = request.lora_scale > 0.0f ? request.lora_scale : 1.0f;
      params.lora_adapters.push_back(lora);
    }

    // Default is 8192 MiB — way too much for mobile/embedded.
    // 0 = disable host-memory prompt caching entirely.
    // The KV cache in the llama_context still handles prompt reuse;
    // this only controls the EXTRA host-RAM cache from PR #16391.
    params.cache_ram_mib = 0;

    // Load-time tuning overrides (KV cache dtype, host prompt cache, n-gram
    // self-speculation). Applied before the server is fetched/created so
    // get_or_create() sees the values its cache key is compared against.
    const auto runtime_overrides = parse_overrides(
        request.runtime_overrides_json, "runtime_overrides_json",
        request.dart_logger);
    apply_runtime_overrides(runtime_overrides, params, request.dart_logger);

#if TARGET_IPHONE_SIMULATOR
    params.n_gpu_layers = 0;
#else
    params.n_gpu_layers = request.num_gpu_layers;
#endif

    if (request.model_mmproj_path && strlen(request.model_mmproj_path) > 0)
      params.mmproj.path = request.model_mmproj_path;

    // Multi-Token Prediction (MTP) speculative decoding. When a drafter/
    // assistant GGUF is supplied (e.g. gemma-4-31B-it-assistant), wire it into
    // common_params.speculative; server_context handles the shared-KV draft
    // loop. Leave draft cache_type_k/v at their F16 defaults — Q8 KV cache was
    // shown to collapse MTP draft acceptance (llama.cpp#23398).
    if (request.draft_model_path && strlen(request.draft_model_path) > 0) {
      params.speculative.draft.mparams.path = request.draft_model_path;
      params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
      params.speculative.draft.n_max =
          request.draft_n_max > 0 ? request.draft_n_max : 3;
      if (request.draft_p_min >= 0.0f) {
        params.speculative.draft.p_min = request.draft_p_min;
      }
      log_message("[fllama] MTP draft enabled: n_max=" +
                      std::to_string(params.speculative.draft.n_max) +
                      ", p_min=" +
                      std::to_string(params.speculative.draft.p_min),
                  request.dart_logger);
      // Offload the (small) drafter alongside the target model.
      params.speculative.draft.n_gpu_layers = params.n_gpu_layers;
    }

    // ── 2. Get or create server_context ───────────────────────────────

    auto *srv = g_mgr.get_or_create(
        request.model_path, params, request.dart_logger);
    if (!srv || !srv->srv_ctx) {
      const std::string reason = fllama_take_last_llama_error();
      emit_inference_callback(
          callback,
          reason.empty()
              ? std::string("Error: Failed to create inference context")
              : "Error: Failed to create inference context: " + reason,
          "", true);
      return;
    }
    // RAII — release when we leave scope.
    struct Guard {
      ServerManager &m; std::string p;
      ~Guard() { m.release(p); }
    } guard{g_mgr, request.model_path};

    log_message("[fllama] Model ready (" +
                    std::to_string(ggml_time_ms() - t0) + " ms)",
                request.dart_logger);

    // ── 3. Build the prompt ───────────────────────────────────────────

    std::string prompt = request.input ? request.input : "";
    common_chat_parser_params parser_params;
    common_chat_params chat_params;
    bool is_oai = false;

    if (request.openai_request_json_string) {
      is_oai = true;
      try {
        auto body = nlohmann::ordered_json::parse(
            request.openai_request_json_string);

        std::string jinja_tmpl;
        if (body.contains("jinja_template") &&
            body["jinja_template"].is_string()) {
          jinja_tmpl = body["jinja_template"].get<std::string>();
          body.erase("jinja_template");
        }

        auto *lctx  = srv->srv_ctx->get_llama_context();
        auto *model  = llama_get_model(lctx);
        auto  tmpls  = common_chat_templates_init(model, jinja_tmpl);

        try {
          std::map<std::string, std::string> empty;
          common_chat_format_example(tmpls.get(), true, empty);
        } catch (...) {
          tmpls = common_chat_templates_init(model, "chatml");
        }

        if (body.contains("messages") && body["messages"].is_array()) {
          common_chat_templates_inputs inputs;
          inputs.use_jinja = true;
          inputs.add_generation_prompt = true;
          inputs.messages =
              common_chat_msgs_parse_oaicompat(body["messages"]);

          // Default to automatic reasoning extraction for modern reasoning/
          // channel-based templates (Qwen, GPT-OSS/Harmony, etc). Allow the
          // request body to override explicitly.
          inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
          inputs.enable_thinking = true;
          if (body.contains("reasoning_format") && body["reasoning_format"].is_string()) {
            inputs.reasoning_format = common_reasoning_format_from_name(
                body["reasoning_format"].get<std::string>());
          }

          if (body.contains("tools")) {
            inputs.tools =
                common_chat_tools_parse_oaicompat(body["tools"]);
            inputs.tool_choice =
                body.contains("tool_choice")
                    ? common_chat_tool_choice_parse_oaicompat(
                          body["tool_choice"]
                              .template get<std::string>())
                    : COMMON_CHAT_TOOL_CHOICE_AUTO;
          }

          auto result =
              common_chat_templates_apply(tmpls.get(), inputs);
          chat_params = result;
          prompt = result.prompt;
          parser_params = common_chat_parser_params(result);
          parser_params.reasoning_format = inputs.reasoning_format;
          parser_params.reasoning_in_content =
              (inputs.reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY);
          if (!result.parser.empty()) {
            parser_params.parser.load(result.parser);
          }

          if constexpr (kLogVerboseDebugMessages) {
            log_message("[fllama] fllama inputs.reasoning_format=" +
                            std::string(common_reasoning_format_name(inputs.reasoning_format)),
                        request.dart_logger);
            log_message("[fllama] Chat format: " +
                            std::string(common_chat_format_name(
                                result.format)),
                        request.dart_logger);
            log_message("[fllama] PROMPT (" +
                            std::to_string(prompt.size()) + " chars):\n" +
                            prompt,
                        request.dart_logger);
          }
        }
      } catch (const std::exception &e) {
        std::string msg = "Error: OAI parse error: " + std::string(e.what());
        log_message(std::string("[fllama] OAI parse error: ") + e.what(),
                    request.dart_logger);
        emit_inference_callback(callback, msg, "", true);
        g_mgr.clear_cancel(request.request_id);
        g_mgr.unregister_request_thread(request.request_id);
        return;
      }
    }

    // ── 4. Multimodal — extract base64 → raw bytes ───────────────────

    std::vector<raw_buffer> files;
    if (fllama_prompt_contains_image(prompt)) {
      const char * media_marker = get_media_marker();
      auto img = fllama_extract_images(prompt, media_marker);
      prompt = std::move(img.text_with_markers);
      for (auto &fb : img.file_bytes)
        files.push_back(std::move(fb));
      log_message("[fllama] Extracted " +
                      std::to_string(files.size()) + " image(s) using marker " +
                      media_marker,
                  request.dart_logger);
    }

    // ── 5. Create & post the server task ──────────────────────────────

    auto reader = srv->srv_ctx->get_response_reader();

    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.id         = reader.get_new_id();
    task.index      = 0;
    task.cli        = true;
    task.cli_prompt = prompt;
    task.cli_files  = std::move(files);

    task.params.stream       = true;
    task.params.cache_prompt = true;
    task.params.n_predict    = request.max_tokens;
    task.params.sampling.temp           = request.temperature;
    task.params.sampling.top_p          = request.top_p;
    task.params.sampling.penalty_freq   = request.penalty_freq;
    task.params.sampling.penalty_repeat = request.penalty_repeat;

    if (is_oai) {
      if (!chat_params.grammar.empty()) {
        task.params.sampling.grammar = common_grammar(
            COMMON_GRAMMAR_TYPE_TOOL_CALLS,
            chat_params.grammar);
      }
      task.params.sampling.grammar_lazy = chat_params.grammar_lazy;
      task.params.sampling.grammar_triggers = chat_params.grammar_triggers;
      task.params.sampling.generation_prompt = chat_params.generation_prompt;
      task.params.antiprompt = chat_params.additional_stops;

      auto *lctx = srv->srv_ctx->get_llama_context();
      auto *model = llama_get_model(lctx);
      auto *vocab = llama_model_get_vocab(model);
      for (const auto &preserved_token : chat_params.preserved_tokens) {
        auto ids = common_tokenize(vocab, preserved_token, false, true);
        if (ids.size() == 1) {
          task.params.sampling.preserved_tokens.insert(ids[0]);
        }
      }
    }

    std::random_device rd;
    task.params.sampling.seed = rd();

    // Per-request tuning overrides. Applied last so they win over everything
    // above — including the random seed, which a benchmark wants to pin.
    {
      const auto ov = parse_overrides(request.request_overrides_json,
                                      "request_overrides_json",
                                      request.dart_logger);
      auto &smp = task.params.sampling;
      std::vector<std::string> known;
      apply_override(ov, "top_k", smp.top_k, &known);
      apply_override(ov, "min_p", smp.min_p, &known);
      apply_override(ov, "typ_p", smp.typ_p, &known);
      apply_override(ov, "top_n_sigma", smp.top_n_sigma, &known);
      // penalty_repeat / penalty_freq are set from the request fields above;
      // an override here wins. Both are listed because fllama's OpenAI layer
      // maps presencePenalty (default 1.1) onto llama.cpp's penalty_repeat,
      // whose own default is 1.0 — so every request carries a repetition
      // penalty unless the caller says otherwise, and turning that off has to
      // be expressible.
      apply_override(ov, "penalty_repeat", smp.penalty_repeat, &known);
      apply_override(ov, "penalty_freq", smp.penalty_freq, &known);
      apply_override(ov, "penalty_last_n", smp.penalty_last_n, &known);
      apply_override(ov, "penalty_present", smp.penalty_present, &known);
      apply_override(ov, "dry_multiplier", smp.dry_multiplier, &known);
      apply_override(ov, "dry_base", smp.dry_base, &known);
      apply_override(ov, "dry_allowed_length", smp.dry_allowed_length, &known);
      apply_override(ov, "seed", smp.seed, &known);
      apply_override(ov, "n_cache_reuse", task.params.n_cache_reuse, &known);

      // Cache reuse needs a context whose KV cache can be shifted; the server
      // silently ignores it otherwise (and always for multimodal). Say so —
      // a benchmark that can't tell "disabled" from "no benefit" will write
      // the wrong conclusion down.
      if (task.params.n_cache_reuse > 0) {
        auto *lctx = srv->srv_ctx->get_llama_context();
        const bool can_shift = llama_memory_can_shift(llama_get_memory(lctx));
        log_message(std::string("[fllama] n_cache_reuse=") +
                        std::to_string(task.params.n_cache_reuse) +
                        (can_shift ? " (active)"
                                   : " IGNORED: this context cannot shift its "
                                     "KV cache"),
                    request.dart_logger);
      }
      for (const char *k : {"reasoning_budget_tokens", "reasoning_budget_start_tag",
                            "reasoning_budget_end_tag", "reasoning_budget_message"}) {
        known.push_back(k);
      }
      warn_unknown_keys(ov, known, "request_overrides_json", request.dart_logger);

      // Reasoning budget: cap the chain-of-thought at N tokens, then force the
      // closing tag (preceded by an optional message) so the model has to stop
      // thinking and answer. Without it a small reasoning model can spend the
      // whole window thinking and emit no answer at all.
      int32_t rbudget = -1;
      apply_override(ov, "reasoning_budget_tokens", rbudget);
      if (rbudget >= 0) {
        std::string start_tag = "<think>", end_tag = "</think>", msg;
        apply_override(ov, "reasoning_budget_start_tag", start_tag);
        apply_override(ov, "reasoning_budget_end_tag", end_tag);
        apply_override(ov, "reasoning_budget_message", msg);

        auto *lctx = srv->srv_ctx->get_llama_context();
        auto *vocab = llama_model_get_vocab(llama_get_model(lctx));
        smp.reasoning_budget_tokens = rbudget;
        smp.reasoning_budget_start = common_tokenize(vocab, start_tag, false, true);
        smp.reasoning_budget_end = common_tokenize(vocab, end_tag, false, true);
        smp.reasoning_budget_forced =
            common_tokenize(vocab, msg + end_tag, false, true);
        smp.reasoning_budget_message = msg;
      }
    }

    if (is_oai) {
      task.params.res_type           = TASK_RESPONSE_TYPE_OAI_CHAT;
      task.params.oaicompat_model    = request.model_path;
      task.params.oaicompat_cmpl_id  = gen_chatcmplid();
      task.params.chat_parser_params = parser_params;
    }

    reader.post_task(std::move(task));

    // ── 6. Read results, invoke callbacks ─────────────────────────────

    int rid = request.request_id;
    auto should_stop = [&] { return g_mgr.is_cancelled(rid); };

    std::string full_content;
    std::string last_json;

    while (reader.has_next()) {
      server_task_result_ptr res;
      try {
        res = reader.next(should_stop);
      } catch (const std::exception &e) {
        // Final parse can fail (e.g. doubled generated_text in update_chat_msg).
        // Log and break — we still have the accumulated text + last good JSON.
        if constexpr (kLogVerboseDebugMessages) {
          log_message(std::string("[fllama] reader.next() threw: ") + e.what(),
                      request.dart_logger);
        }
        break;
      }
      if (!res) break;

      if (res->is_error()) {
        auto ej = res->to_json();
        std::string msg = ej.contains("message")
                              ? ej["message"].get<std::string>()
                              : ej.dump();
        if (fllama_error_requires_backend_recreation(msg)) {
          log_message("[fllama] Backend compute error; marking context "
                      "unhealthy so the next request recreates it",
                      request.dart_logger);
          g_mgr.mark_unhealthy(request.model_path);
        }
        emit_inference_callback(callback, msg, "", true);
        g_mgr.clear_cancel(rid);
        g_mgr.unregister_request_thread(rid);
        return;
      }

      auto *partial =
          dynamic_cast<server_task_result_cmpl_partial *>(res.get());
      if (partial) {
        full_content += partial->content;
        if constexpr (kLogVerboseDebugMessages) {
          log_message("[fllama] token: \"" + partial->content +
                      "\"  cumulative(" + std::to_string(full_content.size()) +
                      " chars)",
                      request.dart_logger);
        }
        auto j = res->to_json();
        if (!j.is_null()) {
          last_json = j.dump();
          emit_inference_callback(callback, full_content, last_json, false);
        }
        continue;
      }

      auto *final_r =
          dynamic_cast<server_task_result_cmpl_final *>(res.get());
      if (final_r) {
        // Keep accumulated full_content — final_r->content can be
        // empty or corrupted for tool-call / reasoning completions.
        try {
          auto j = res->to_json();
          last_json = j.is_null() ? "" : j.dump();
          if constexpr (kLogVerboseDebugMessages) {
            log_message("[fllama] final to_json() is_null=" +
                            std::to_string(j.is_null()) +
                            " type=" + std::to_string((int)j.type()) +
                            " size=" + std::to_string(j.size()) +
                            " dump=" + last_json.substr(0, 200),
                        request.dart_logger);
          }
        } catch (const std::exception &e) {
          if constexpr (kLogVerboseDebugMessages) {
            log_message(std::string("[fllama] final to_json() THREW: ") + e.what(),
                        request.dart_logger);
          }
          last_json = "";
        }
        if constexpr (kLogVerboseDebugMessages) {
          log_message("[fllama] final_r->content(" +
                          std::to_string(final_r->content.size()) +
                          ")=\"" + final_r->content.substr(0, 100) + "\"",
                      request.dart_logger);
        }
        emit_inference_callback(callback, full_content, last_json, true);

        log_message("[fllama] Done. " +
                        std::to_string(final_r->n_decoded) + " tok, " +
                        std::to_string(ggml_time_ms() - t0) + " ms",
                    request.dart_logger);
        g_mgr.clear_cancel(rid);
        g_mgr.unregister_request_thread(rid);
        return;
      }
    }

    // Cancelled or exhausted without final result.
    emit_inference_callback(callback, full_content, last_json, true);
    g_mgr.clear_cancel(rid);
    g_mgr.unregister_request_thread(rid);

  } catch (const std::exception &e) {
    std::string msg = "Error: " + std::string(e.what());
    emit_inference_callback(callback, msg, "", true);
    g_mgr.clear_cancel(request.request_id);
    g_mgr.unregister_request_thread(request.request_id);
  } catch (...) {
    emit_inference_callback(callback, "Error: Unknown exception", "", true);
    g_mgr.clear_cancel(request.request_id);
    g_mgr.unregister_request_thread(request.request_id);
  }
}

// ── FFI entry points ─────────────────────────────────────────────────────────

extern "C" {

EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT int fllama_get_gpu_device_count(void) {
  return static_cast<int>(fllama_get_gpu_devices().size());
}

EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT int fllama_get_gpu_memory_info(
    int gpu_index,
    struct fllama_gpu_memory_info * out_info) {
  if (out_info == nullptr) {
    return 1;
  }

  std::memset(out_info, 0, sizeof(*out_info));

  if (gpu_index < 0) {
    return 2;
  }

  auto devices = fllama_get_gpu_devices();
  if (static_cast<size_t>(gpu_index) >= devices.size()) {
    return 3;
  }

  auto * dev = devices[static_cast<size_t>(gpu_index)];
  ggml_backend_dev_props props{};
  ggml_backend_dev_get_props(dev, &props);

  size_t total = props.memory_total;
  size_t free = props.memory_free;

  // Metal reports free as recommendedMaxWorkingSetSize - currentAllocatedSize.
  // If currentAllocatedSize exceeds the recommendation, the backend can
  // underflow the unsigned subtraction. Clamp that to zero here.
  if (total == 0) {
    free = 0;
  } else if (free > total) {
    free = 0;
  }

  out_info->device_index = gpu_index;
  out_info->total_bytes = static_cast<uint64_t>(total);
  out_info->free_bytes = static_cast<uint64_t>(free);
  fllama_copy_cstr(out_info->name, sizeof(out_info->name), props.name);
  fllama_copy_cstr(
      out_info->description,
      sizeof(out_info->description),
      props.description);
  fllama_copy_cstr(
      out_info->device_id,
      sizeof(out_info->device_id),
      props.device_id);
  return 0;
}

EMSCRIPTEN_KEEPALIVE void fllama_inference(fllama_inference_request request,
                                           fllama_inference_callback callback) {
  int rid = request.request_id;
  std::thread t([request, callback] { run_inference(request, callback); });
  g_mgr.register_request_thread(rid, std::move(t));
}

EMSCRIPTEN_KEEPALIVE void
fllama_inference_sync(fllama_inference_request request,
                      fllama_inference_callback callback) {
  // Synchronous variant — blocks the calling thread.
  run_inference(request, callback);
}

EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT void
fllama_inference_cancel(int request_id) {
  g_mgr.cancel(request_id);
}

EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT void
fllama_evict_idle_servers(const char *except_model_path) {
  g_mgr.evict_idle_except(
      except_model_path ? std::string(except_model_path) : std::string());
}

} // extern "C"
