Gem::Specification.new do |spec|
  spec.name                    = "comb"
  spec.version                 = "0.1.0"
  spec.authors                 = ["Chris McComb"]

  spec.summary                 = %q{A simple, single page Jekyll theme.}
  spec.homepage                = "https://github.com/cmccomb/comb"
  spec.license                 = "MIT"

  spec.metadata["plugin_type"] = "theme"

  spec.files                   = `git ls-files -z`.split("\x0").select do |f|
    f.match(%r{^(assets|_(data|includes|layouts|sass)/|(LICENSE|README)((\.(txt|md|markdown)|$)))}i)
  end

  spec.add_runtime_dependency "jekyll", ">= 3.7", "< 5.0"

  spec.add_development_dependency "bundler"
end
