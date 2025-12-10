using Flux, Tsunami
using Documenter

DocMeta.setdocmeta!(Tsunami, :DocTestSetup, :(using Tsunami, Flux); recursive = true)

prettyurls = get(ENV, "CI", nothing) == "true"
mathengine = MathJax3()
sidebar_sitename = true
assets = ["assets/flux.css"]

# Copy README to index.md and fix image paths for documentation
readme_path = joinpath(@__DIR__, "..", "README.md")
index_path = joinpath(@__DIR__, "src", "index.md")
readme_content = read(readme_path, String)
# Replace 
# <img align="right" width="200px" src="https://raw.githubusercontent.com/CarloLucibello/Tsunami.jl/main/docs/src/assets/the_great_wave.jpg">
# with
# ![Great Wave](src/assets/the_great_wave.jpg)
readme_content = replace(
    readme_content,
    "<img align=\"right\" width=\"200px\" src=\"https://raw.githubusercontent.com/CarloLucibello/Tsunami.jl/main/docs/src/assets/the_great_wave.jpg\">" =>
    "![Great Wave](https://raw.githubusercontent.com/CarloLucibello/Tsunami.jl/main/docs/src/assets/the_great_wave.jpg)"
)

write(index_path, readme_content)

makedocs(;
    modules = [Tsunami],
    format = Documenter.HTML(; mathengine, prettyurls, assets, sidebar_sitename),
    sitename = "Tsunami.jl",
    pages = [
        "Home" => "index.md",
        
        # "Get Started" => [] # TODO These are the first things you should read.
        # "Tutorials" => [] # TODO These walk you through various tasks. It's fine if they overlap quite a lot.

        "How-To Guides" => "guides.md",
        
        "API Reference" => [
            # This essentially collects docstrings, with a bit of introduction.
            "Callbacks" => "api/callbacks.md",
            "FluxModule" => "api/fluxmodule.md",
            "Foil" => "api/foil.md",
            "Hooks" => "api/hooks.md",
            "Logging" => "api/logging.md",
            "Trainer" => "api/trainer.md",
            "Utils" => "api/utils.md",
        ],
    ]
)

rm(joinpath(@__DIR__, "src", "index.md"), force=true)

deploydocs(repo = "github.com/CarloLucibello/Tsunami.jl.git")
