########################################################################
# Goal: install relevant packages and produce all figures in the paper
# Julia Version 1.5.3
########################################################################

########################################################################
# 1) Install relevant packages
########################################################################

using Pkg

# Install all relevant packages
for package in ["Distributions", "Random", "DataFrames", "Gadfly", "Cairo", "Fontconfig", "LinearAlgebra", "JuMP", "Clp"]
    Pkg.add(package)
end

#Pin packages to version used in the paper
Pkg.pin(name="Distributions",version="0.25.76")
Pkg.pin(name="DataFrames",version="1.3.6")
Pkg.pin(name="Gadfly",version="1.3.4")
Pkg.pin(name="Cairo",version="1.0.5")
Pkg.pin(name="Fontconfig",version="0.4.0")
Pkg.pin(name="JuMP",version="0.21.10")
Pkg.pin(name="Clp",version="0.8.4")

########################################################################
# 2) Run scripts for replication of Figures
########################################################################

#Path to save figures (TO SPECIFY DEPENDING ON USER)
path_figures = ""
#note that if the path to save figures is unspecified, the figures will be saved in the
#working directory by default

#Figure 1 and Figure 2 in main text
include("Identified_Set_Logit_T2.jl")
include("Identified_Set_Logit_T3.jl")
include("Identified_Set_Logit_T4.jl")

include("Identified_Set_Probit_T2.jl")
include("Identified_Set_Probit_T3.jl")
include("Identified_Set_Probit_T4.jl")

#Appendix Figure 1 and Appendix Figure 2
include("Identified_Set_Logit_T2_Approx.jl")
include("Identified_Set_Probit_T2_Approx.jl")
