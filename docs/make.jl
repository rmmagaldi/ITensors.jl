include("settings.jl")

makedocs(; sitename=sitename, settings...)

deploydocs(; repo="github.com/ITensor/ITensors.jl.git")
