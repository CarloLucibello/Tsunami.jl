using Flux, Tsunami
using Documenter

DocMeta.setdocmeta!(Tsunami, :DocTestSetup,
                    :(using Tsunami, Flux);
                    recursive = true)

prettyurls = get(ENV, "CI", nothing) == "true"
mathengine = MathJax3()
sidebar_sitename = true
assets = ["assets/flux.css"]

makedocs(;
         modules = [Tsunami],
         doctest = true,
         clean = true,
         format = Documenter.HTML(; mathengine, prettyurls, assets, sidebar_sitename),
         sitename = "Tsunami.jl",
         pages = ["Home" => "index.md",
             "FluxModule" => "fluxmodule.md",
             "Trainer" => "trainer.md",
         ])

deploydocs(repo = "github.com/CarloLucibello/Tsunami.jl.git")
