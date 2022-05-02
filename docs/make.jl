using BBModels
using Documenter

DocMeta.setdocmeta!(BBModels, :DocTestSetup, :(using BBModels); recursive = true)

makedocs(;
  modules = [BBModels],
  doctest = true,
  linkcheck = false,
  strict = false,
  authors = "Abel Soares Siqueira <abel.s.siqueira@gmail.com> and contributors",
  repo = "https://github.com/JuliaSmoothOptimizers/BBModels.jl/blob/{commit}{path}#{line}",
  sitename = "BBModels.jl",
  format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://JuliaSmoothOptimizers.github.io/BBModels.jl",
    assets = ["assets/style.css"],
  ),
  pages = ["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(;
  repo = "github.com/JuliaSmoothOptimizers/BBModels.jl",
  push_preview = true,
  devbranch = "main",
)
