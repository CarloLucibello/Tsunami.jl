using Flux, Tsunami
using Documenter

DocMeta.setdocmeta!(Tsunami, :DocTestSetup, :(using Tsunami, Flux); recursive = true)

prettyurls = get(ENV, "CI", nothing) == "true"
mathengine = MathJax3()
sidebar_sitename = true
assets = ["assets/flux.css"]

makedocs(;
    modules = [Tsunami],
    doctest = true,
    checkdocs = :exports,
    format = Documenter.HTML(; mathengine, prettyurls, assets, sidebar_sitename),
    sitename = "Tsunami.jl",
    pages = [
        "Get Started" => "index.md",
        
        "Guides" => "guides.md",
        
        # "Tutorials" => [] # TODO These walk you through various tasks. It's fine if they overlap quite a lot.

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

deploydocs(repo = "github.com/CarloLucibello/Tsunami.jl.git")
