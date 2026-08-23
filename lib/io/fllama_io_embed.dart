import 'dart:async';
import 'dart:convert';
import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:fllama/fllama_io.dart';
import 'package:fllama/io/fllama_bindings_generated.dart';

/// Result of [fllamaEmbed]: one vector per input, in input order.
class FllamaEmbedResult {
  const FllamaEmbedResult({required this.embeddings, required this.tokenCount});

  /// One vector per input. L2-normalized unless `normalize: false` was passed,
  /// which is what cosine similarity expects — with unit vectors the cosine of
  /// two embeddings is just their dot product.
  final List<List<double>> embeddings;

  /// Total tokens the model processed. Useful for cost/pacing measurements.
  final int tokenCount;
}

/// Result of [fllamaRerank]: one score per document, in input order.
class FllamaRerankResult {
  const FllamaRerankResult({required this.scores, required this.tokenCount});

  /// Relevance of each document to the query. Higher is more relevant. The
  /// scale is model-specific and NOT a probability — compare scores within one
  /// call, never across models or across calls with different queries.
  final List<double> scores;
  final int tokenCount;
}

int _nextRequestId = 900000000;

/// Embeds [inputs] with the model at [modelPath].
///
/// A model is either an embedder or a chat model: llama.cpp decides at load
/// time whether the context pools or generates. Pass an embedding GGUF here
/// (e.g. BGE-M3, Qwen3-Embedding) — pointing this at a chat model loads a
/// second, pooling copy of it rather than reusing the chat context.
///
/// [pooling] must match how the model was trained; the wrong pooling returns
/// vectors that look fine and rank badly. Most sentence embedders use `mean`;
/// BERT-style models with a [CLS] head use `cls`; decoder-style embedders
/// (Qwen3-Embedding) use `last`.
Future<FllamaEmbedResult> fllamaEmbed({
  required String modelPath,
  required List<String> inputs,
  int contextSize = 2048,
  int numGpuLayers = 99,
  int numThreads = 2,
  String pooling = 'mean',
  bool normalize = true,
}) async {
  final json = jsonEncode({
    'input': inputs,
    'pooling': pooling,
    'normalize': normalize,
  });
  final result = await _embedRaw(
    modelPath: modelPath,
    inputJson: json,
    contextSize: contextSize,
    numGpuLayers: numGpuLayers,
    numThreads: numThreads,
  );
  final raw = (result['embeddings'] as List?) ?? const [];
  return FllamaEmbedResult(
    embeddings: [
      for (final v in raw) [for (final x in v as List) (x as num).toDouble()],
    ],
    tokenCount: (result['n_tokens'] as num?)?.toInt() ?? 0,
  );
}

/// Scores each of [documents] against [query] with a reranking model.
///
/// This is a different model family from [fllamaEmbed]: a reranker reads the
/// query and the document TOGETHER and emits one number, which is why it
/// beats cosine similarity on the same corpus — and why it cannot be indexed
/// ahead of time. The usual shape is retrieve wide with embeddings (or BM25),
/// then rerank the top handful.
Future<FllamaRerankResult> fllamaRerank({
  required String modelPath,
  required String query,
  required List<String> documents,
  int contextSize = 2048,
  int numGpuLayers = 99,
  int numThreads = 2,
}) async {
  final json = jsonEncode({'query': query, 'documents': documents});
  final result = await _embedRaw(
    modelPath: modelPath,
    inputJson: json,
    contextSize: contextSize,
    numGpuLayers: numGpuLayers,
    numThreads: numThreads,
  );
  final raw = (result['scores'] as List?) ?? const [];
  return FllamaRerankResult(
    scores: [for (final x in raw) (x as num).toDouble()],
    tokenCount: (result['n_tokens'] as num?)?.toInt() ?? 0,
  );
}

Future<Map<String, dynamic>> _embedRaw({
  required String modelPath,
  required String inputJson,
  required int contextSize,
  required int numGpuLayers,
  required int numThreads,
}) {
  final completer = Completer<Map<String, dynamic>>();
  final requestPtr = calloc<fllama_embed_request>();
  final modelPtr = modelPath.toNativeUtf8().cast<Char>();
  final inputPtr = inputJson.toNativeUtf8().cast<Char>();

  // Native calls back from its own thread; a listener callable hands the
  // result to this isolate's event loop. It must outlive the native call, so
  // it is closed inside the callback rather than in a finally block.
  late final NativeCallable<
    Void Function(Pointer<Char>, Pointer<Char>)
  > callback;

  void onDone(Pointer<Char> resultJson, Pointer<Char> error) {
    try {
      if (error != nullptr) {
        completer.completeError(
          Exception('fllama_embed: ${error.cast<Utf8>().toDartString()}'),
        );
      } else if (resultJson == nullptr) {
        completer.completeError(Exception('fllama_embed: no result'));
      } else {
        completer.complete(
          jsonDecode(resultJson.cast<Utf8>().toDartString())
              as Map<String, dynamic>,
        );
      }
    } finally {
      calloc.free(requestPtr);
      malloc.free(modelPtr);
      malloc.free(inputPtr);
      callback.close();
    }
  }

  callback = NativeCallable<Void Function(Pointer<Char>, Pointer<Char>)>.listener(
    onDone,
  );

  final req = requestPtr.ref;
  req.request_id = _nextRequestId++;
  req.context_size = contextSize;
  req.model_path = modelPtr;
  req.num_gpu_layers = numGpuLayers;
  req.num_threads = numThreads;
  req.input_json = inputPtr;

  fllamaBindings.fllama_embed(req, callback.nativeFunction);
  return completer.future;
}
