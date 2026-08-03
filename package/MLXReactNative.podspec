require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "MLXReactNative"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]

  s.platforms    = { :ios => 26.0, :visionos => 1.0 }
  s.source       = { :git => "https://github.com/corasan/react-native-nitro-mlx.git", :tag => "#{s.version}" }

  s.source_files = [
    # Implementation (Swift)
    "ios/Sources/**/*.{swift}",
    # Autolinking/Registration (Objective-C++)
    "ios/**/*.{m,mm}",
    # Implementation (C++ objects)
    "cpp/**/*.{hpp,cpp}",
  ]

  spm_dependency(s,
    url: "https://github.com/ml-explore/mlx-swift-lm.git",
    requirement: {kind: "upToNextMinorVersion", minimumVersion: "3.31.3"},
    products: ["MLXLLM", "MLXLMCommon", "MLXEmbedders"]
  )

  spm_dependency(s,
    url: "https://github.com/Blaizzy/mlx-audio-swift.git",
    # Pinned to a revision, not `main`: a clean prebuild re-resolves SPM, so tracking a
    # branch lets upstream changes break the build with no local change. Bump deliberately.
    requirement: {kind: "revision", revision: "4266f988d170a83017d1e82e2e4654602f277f1d"},
    products: ["MLXAudioTTS", "MLXAudioSTT", "MLXAudioCore"]
  )

  s.pod_target_xcconfig = {
    # C++ compiler flags, mainly for folly.
    "GCC_PREPROCESSOR_DEFINITIONS" => "$(inherited) FOLLY_NO_CONFIG FOLLY_CFG_NO_COROUTINES"
  }

  load 'nitrogen/generated/ios/MLXReactNative+autolinking.rb'
  add_nitrogen_files(s)

  s.dependency 'React-jsi'
  s.dependency 'React-callinvoker'
  install_modules_dependencies(s)
end
