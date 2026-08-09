import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:path/path.dart' as p;

/// Source extensions that affect the native fllama build.
const sourceExtensions = <String>{
  '.c',
  '.cc',
  '.cpp',
  '.cxx',
  '.h',
  '.hh',
  '.hpp',
  '.hxx',
  '.m',
  '.mm',
  '.metal',
  '.cmake',
};

/// Additional source basenames that affect the native fllama build.
const sourceBasenames = <String>{'CMakeLists.txt'};

/// A path-independent fingerprint of one native build input.
class SourceFileFingerprint {
  const SourceFileFingerprint(
    this.relPath,
    this.absoluteUri,
    this.size,
    this.contentDigest,
  );

  final String relPath;
  final Uri absoluteUri;
  final int size;
  final Digest contentDigest;
}

/// Collects native build inputs and hashes their contents.
///
/// Content digests deliberately replace mtimes here. Git checkouts assign new
/// mtimes on every fresh CI machine, which made identical source trees produce
/// different cache keys and prevented cross-build cache hits.
Future<List<SourceFileFingerprint>> collectSourceFiles(Uri sourceDir) async {
  final srcDir = Directory.fromUri(sourceDir);
  final basePath = srcDir.path;
  final candidates = <File>[];

  await for (final entity in srcDir.list(recursive: true, followLinks: false)) {
    if (entity is! File) continue;
    final path = entity.path;
    final ext = p.extension(path).toLowerCase();
    final basename = p.basename(path);
    if (!sourceExtensions.contains(ext) &&
        !sourceBasenames.contains(basename)) {
      continue;
    }
    // Skip anything inside a build/ directory — output, not input. (Only
    // relevant if someone did a local cmake build by hand.)
    if (p.split(p.relative(path, from: basePath)).contains('build')) {
      continue;
    }
    candidates.add(entity);
  }

  candidates.sort(
    (a, b) => p
        .relative(a.path, from: basePath)
        .compareTo(p.relative(b.path, from: basePath)),
  );

  final files = <SourceFileFingerprint>[];
  for (final file in candidates) {
    final stat = await file.stat();
    final digest = await sha256.bind(file.openRead()).single;
    files.add(
      SourceFileFingerprint(
        p.relative(file.path, from: basePath),
        file.uri,
        stat.size,
        digest,
      ),
    );
  }
  return files;
}

/// Computes a stable key for a native fllama build.
String computeBuildKey({
  required String os,
  required String arch,
  required String? iosSdk,
  required Map<String, String> defines,
  required List<SourceFileFingerprint> sourceFiles,
}) {
  final buffer = StringBuffer();
  // v3 folds the iOS SDK into the key. v2 (content digests) could not tell an
  // iphoneos build from an iphonesimulator one — both are arm64 with the same
  // defines — so whichever built first won and a simulator dylib could end up
  // in a device archive, failing App Store validation with a 409.
  buffer.writeln('v3');
  buffer.writeln('os=$os');
  buffer.writeln('arch=$arch');
  // Device vs simulator produce arm64 slices for different platforms; keep
  // their caches distinct (null for every non-iOS target — no key change).
  if (iosSdk != null) buffer.writeln('iosSdk=$iosSdk');

  final sortedDefines = defines.entries.toList()
    ..sort((a, b) => a.key.compareTo(b.key));
  for (final entry in sortedDefines) {
    buffer.writeln('D:${entry.key}=${entry.value}');
  }
  for (final file in sourceFiles) {
    buffer.writeln('F:${file.relPath}|${file.size}|${file.contentDigest}');
  }

  final digest = sha256.convert(utf8.encode(buffer.toString()));
  // 16 hex chars = 64 bits. For ~10^5 distinct cache entries the
  // birthday-collision probability is ~2.7e-10 — effectively zero.
  return digest.toString().substring(0, 16);
}
