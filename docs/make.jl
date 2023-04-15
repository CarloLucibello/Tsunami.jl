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
         pages = [
              "Get Started" => "index.md",
              
              "Guides" => "guides.md",
       
              "API Reference" => [
                     # This essentially collects docstrings, with a bit of introduction.
                     "Callbacks" => "callbacks.md",
                     "FluxModule" => "fluxmodule.md",
                     "Hooks" => "hooks.md",
                     "Logging" => "logging.md",
                     "Trainer" => "trainer.md",
                     "Utils" => "utils.md",
              ],
                # "Tutorials" => [] # TODO These walk you through various tasks. It's fine if they overlap quite a lot.
             
         ])

deploydocs(repo = "github.com/CarloLucibello/Tsunami.jl.git")
