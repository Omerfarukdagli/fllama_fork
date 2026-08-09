import 'dart:io';

import 'package:test/test.dart';

import '../hook/cache_key.dart';

void main() {
  late Directory firstSourceDirectory;
  late Directory secondSourceDirectory;

  setUp(() async {
    firstSourceDirectory = await Directory.systemTemp.createTemp(
      'fllama_cache_key_first_',
    );
    secondSourceDirectory = await Directory.systemTemp.createTemp(
      'fllama_cache_key_second_',
    );
  });

  tearDown(() async {
    await firstSourceDirectory.delete(recursive: true);
    await secondSourceDirectory.delete(recursive: true);
  });

  Future<String> keyFor(Directory directory, {String? iosSdk}) async {
    return computeBuildKey(
      os: iosSdk == null ? 'macos' : 'ios',
      arch: 'arm64',
      iosSdk: iosSdk,
      defines: const {'CMAKE_BUILD_TYPE': 'Release'},
      sourceFiles: await collectSourceFiles(directory.uri),
    );
  }

  test('is stable when checkout path and source mtimes differ', () async {
    final first = File('${firstSourceDirectory.path}/fllama.cpp');
    final second = File('${secondSourceDirectory.path}/fllama.cpp');
    await first.writeAsString('int fllama() { return 1; }\n');
    await second.writeAsString('int fllama() { return 1; }\n');
    await first.setLastModified(DateTime.utc(2024));
    await second.setLastModified(DateTime.utc(2026));

    expect(
      await keyFor(firstSourceDirectory),
      await keyFor(secondSourceDirectory),
    );
  });

  test('changes when source contents change without changing size', () async {
    final source = File('${firstSourceDirectory.path}/fllama.cpp');
    await source.writeAsString('int fllama() { return 1; }\n');
    final originalKey = await keyFor(firstSourceDirectory);

    await source.writeAsString('int fllama() { return 2; }\n');

    expect(await keyFor(firstSourceDirectory), isNot(originalKey));
  });

  // Regression: iphoneos and iphonesimulator both compile arm64 with identical
  // defines, so a key without the SDK collides. Whichever built first won, and
  // a simulator dylib reused for a device archive fails App Store validation
  // ("references an unsupported platform in the arm64 slice") with a 409.
  test('separates iOS device and simulator builds', () async {
    await File('${firstSourceDirectory.path}/fllama.cpp')
        .writeAsString('int fllama() { return 1; }\n');

    final deviceKey = await keyFor(firstSourceDirectory, iosSdk: 'iphoneos');
    final simulatorKey =
        await keyFor(firstSourceDirectory, iosSdk: 'iphonesimulator');

    expect(deviceKey, isNot(simulatorKey));
  });

  test('leaves non-iOS keys unchanged by the SDK field', () async {
    await File('${firstSourceDirectory.path}/fllama.cpp')
        .writeAsString('int fllama() { return 1; }\n');

    // A null SDK must not write anything into the buffer, so macOS/Android
    // keys stay identical to what a key without the field would produce.
    expect(
      await keyFor(firstSourceDirectory),
      await keyFor(firstSourceDirectory),
    );
  });

  test('ignores generated build directories and unrelated files', () async {
    await File('${firstSourceDirectory.path}/fllama.cpp')
        .writeAsString('int fllama() { return 1; }\n');
    final originalKey = await keyFor(firstSourceDirectory);
    await Directory('${firstSourceDirectory.path}/build').create();
    await File('${firstSourceDirectory.path}/build/generated.cpp')
        .writeAsString('generated output\n');
    await File('${firstSourceDirectory.path}/README.md')
        .writeAsString('documentation\n');

    expect(await keyFor(firstSourceDirectory), originalKey);
  });
}
