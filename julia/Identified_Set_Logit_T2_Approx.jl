#######################################################################################################
# Goal: we compare the true identified set for the logit case with T=2
# with approximations of it. To do so, we approximate the distribution
# of unobserved heterogeneity with discrete distributions with 5,50,500 support points respectively
# Julia Version 1.5.3
#######################################################################################################

# Packages
using Distributions
using Random
using DataFrames
using Gadfly
using Cairo
using Fontconfig
using LinearAlgebra
using JuMP
using Clp

########################################################################
# Functions
########################################################################

#Returns probit CDF at u
function cdfnormal(u)
        return cdf(Normal(0,1),u)
end

#Returns logit CDF at u
function softmax(u)
        return exp(u)/(1+exp(u))
end

#Computes probability of a choice history given alpha and x1
function true_conditional_probability_given_alpha_x1(theta,y2,x2,y1,x1,alpha)
        #covariates X are iid Bernouilli(1/2) independent of everything else
        H_alpha = (softmax(theta*x2+alpha)^y2)*((1-softmax(theta*x2+alpha))^(1-y2))*(1/2)*(softmax(theta*x1+alpha)^y1)*((1-softmax(theta*x1+alpha))^(1-y1))
        return H_alpha
end

#Computes probability of a choice history given x1
function true_conditional_probability_given_x1(theta,y2,x2,y1,x1,alpha_grid,pi_alpha)
        H = 0.0
        for (index,alpha) in enumerate(alpha_grid)
                H_alpha = true_conditional_probability_given_alpha_x1(theta,y2,x2,y1,x1,alpha)
                H += H_alpha*pi_alpha[index]
        end
        return H
end

#Computes probability of a choice history for x1=0 and x1=1
function true_probabilities(theta,alpha_grid,pi_alpha)
        H = zeros(2,2,2,2) #order y2,x2,y1,x1 - 1st entry is for "0" and 2nd is for "1"
        for y2 in 0:1
                for x2 in 0:1
                        for y1 in 0:1
                                for x1 in 0:1
                                        H[y2+1,x2+1,y1+1,x1+1]=true_conditional_probability_given_x1(theta,y2,x2,y1,x1,alpha_grid,pi_alpha)
                                end
                        end
                end
        end

        return H
end

#Computes average partial effect under the assumption that the heterogeneity distribution
#does not depend on x1 as in the DGP below
function true_APE(theta,alpha_grid,pi_alpha)

        APE = 0.0

        for (index,alpha) in enumerate(alpha_grid)
                 APE+=(softmax(theta+alpha)-softmax(alpha))*pi_alpha[index]
        end

        return APE
end

########################################################################
# Computation of the identified set for the predetermined case
########################################################################

#Model primitives
T = 2  #Time periods
K = 31  #Number of support points for distribution of alpha

#Heterogeneity
alpha_max=3.0 #maximum of the support of alpha
alpha_min=-3.0 #minimum of the support of alpha
alpha_grid = collect(alpha_min:(alpha_max-alpha_min)/(K-1):alpha_max)   #support points for alpha
pi_alpha=zeros(K)

for k in 1:K
        if k ==1
                pi_alpha[k] = cdfnormal((alpha_grid[k]+alpha_grid[k+1])/2)
        elseif k>1 && k<K
                pi_alpha[k] = cdfnormal((alpha_grid[k]+alpha_grid[k+1])/2)-cdfnormal((alpha_grid[k]+alpha_grid[k-1])/2)
        elseif k==K
                pi_alpha[k] = 1-cdfnormal((alpha_grid[k]+alpha_grid[k-1])/2)
        end
end

#We will compute the identified set for theta ranging from -1 to 1 by increments of 0.1
theta_true_grid = collect(-1.0:0.1:1.0)
theta_grid =  collect(-1.3:0.01:1.3) #grid for potential theta values
N_theta = length(theta_true_grid)
APE_true_grid = map(theta_true->true_APE(theta_true,alpha_grid,pi_alpha),theta_true_grid)

#Store output into a DataFrame
results = DataFrame(
theta_true = theta_true_grid,
theta_set_max =zeros(N_theta),theta_set_min = zeros(N_theta),
theta_set_max_approx5 =zeros(N_theta),theta_set_min_approx5 = zeros(N_theta),
theta_set_max_approx50 =zeros(N_theta),theta_set_min_approx50 = zeros(N_theta),
theta_set_max_approx500 =zeros(N_theta),theta_set_min_approx500 = zeros(N_theta),
APE_true = APE_true_grid,
APE_set_max_upper_bound = zeros(N_theta), APE_set_min_lower_bound =  zeros(N_theta),
APE_set_max_upper_bound_approx5 = zeros(N_theta), APE_set_min_lower_bound_approx5 =  zeros(N_theta),
APE_set_max_upper_bound_approx50 = zeros(N_theta), APE_set_min_lower_bound_approx50 =  zeros(N_theta),
APE_set_max_upper_bound_approx500 = zeros(N_theta), APE_set_min_lower_bound_approx500 =  zeros(N_theta)
)

#### Predetermined case

#1) True identified set where we know the support points of the heterogeneity distribution

for (index,theta_true) in enumerate(theta_true_grid)

        #Vector of true probabilities
        H = true_probabilities(theta_true,alpha_grid,pi_alpha)

        #Identified set for predetermined case
        theta_set = []
        APE_set_max = [] #collects all upper bounds on APE for each theta in the identified set
        APE_set_min =[] #collects all lower bounds on APE for each theta in the identified set

        for theta in theta_grid
                m = Model(Clp.Optimizer) #set up optimization and sets solver
                set_optimizer_attribute(m,"SolveType",1)
                set_silent(m) #prevent solver from showing all outputs

                #1) Declare variables

                #Indexing convention:
                #For x,y: 1st entry is "0", 2nd entry is "1"
                #For alpha: indexing follows alpha_grid

                #Order of dimensions: x2,y1,alpha,x1
                @variable(m, psi[1:2,1:2,1:K,1:2] >= 0)

                #2) Declare linear programming constraints
                constraint = Dict()

                #Constraint for psi to be positive
                constraint[1] = @constraint(m, psi[:,:,:,:] .>= 0)

                #Contraints for psi(;x1) to be a probabilitiy for x1=0,1
                constraint[2] = @constraint(m, 1-sum(psi[:,:,:,1]) == 0)
                constraint[3] = @constraint(m, 1-sum(psi[:,:,:,2]) == 0)

                counter=4
                #Constraints for psi to be consistent with the data
                for y2 in 0:1
                        for y1 in 0:1
                                for x2 in 0:1
                                        for x1 in 0:1
                                        temp = map(alpha-> (softmax(theta*x2+alpha)^y2)*(1-softmax(theta*x2+alpha))^(1-y2),alpha_grid)
                                        constraint[counter]=@constraint(m, H[y2+1,x2+1,y1+1,x1+1]-temp'psi[x2+1,y1+1,:,x1+1]==0)
                                        counter+=1
                                        end
                                end
                        end
                end

                #Constraints for psi to be consistent with outcome distribution in period 1
                for x1 in 0:1
                        for y1 in 0:1
                                for (index,alpha) in enumerate(alpha_grid)
                                        temp = (softmax(theta*x1+alpha)^y1)*(1-softmax(theta*x1+alpha))^(1-y1)
                                        constraint[counter]=@constraint(m,sum(psi[:,y1+1,index,x1+1])-temp*sum(psi[:,:,index,x1+1])==0)
                                        counter+=1
                                end
                        end
                end

                #3) Declare linear programming objective
                @objective(m, Min,0.5*sum(psi[:,:,:,1])+0.5*sum(psi[:,:,:,2]))

                #4) Solving the optimization problem
                JuMP.optimize!(m)

                #5) Add theta to the identified set if the problem is feasible
                # i.e if the constraints of the identified set are verified
                # and add the associated bounds for average partial effects
                if string(termination_status(m))=="OPTIMAL"
                        push!(theta_set,theta)

                        #Upper bound for average partial effects associated to theta
                        @objective(m,Max,sum(map(k->0.5*sum(psi[:,:,k,:])*(softmax(theta+alpha_grid[k])-softmax(alpha_grid[k])),collect(1:K))))
                        JuMP.optimize!(m)
                        push!(APE_set_max,objective_value(m))

                        #Lower bound for average partial effects associated to theta
                        @objective(m,Min,sum(map(k->0.5*sum(psi[:,:,k,:])*(softmax(theta+alpha_grid[k])-softmax(alpha_grid[k])),collect(1:K))))
                        JuMP.optimize!(m)
                        push!(APE_set_min,objective_value(m))

                end

        end

        #Report length of the identified set
        theta_set_max = maximum(theta_set)
        theta_set_min = minimum(theta_set)

        results[index,:theta_set_max]=theta_set_max
        results[index,:theta_set_min]=theta_set_min

        #Report global bounds on average partial effects
        APE_set_max_upper_bound = maximum(APE_set_max)
        APE_set_min_lower_bound = minimum(APE_set_min)

        results[index,:APE_set_max_upper_bound]=APE_set_max_upper_bound
        results[index,:APE_set_min_lower_bound]=APE_set_min_lower_bound
end

#2) Approximated identified set: we approximate the distribution of unobserved heterogeneity by a
# discrete distribution with 5,50,500 points of support

for K_approx in [5,50,500]

        alpha_grid_approx=map(x->quantile(Normal(0,1),x),collect(0:1/(K_approx-1):1)) #the support points are placed like the quantiles of a normal
        alpha_grid_approx[1]=-10 #replace -infty with -10
        alpha_grid_approx[K_approx]=+10 #replace +infty with +10

        for (index,theta_true) in enumerate(theta_true_grid)

        #Vector of true probabilities
        H = true_probabilities(theta_true,alpha_grid,pi_alpha)

        #Approximated identified set for predetermined case
        theta_set_approx = []
        APE_set_max_approx = [] #collects all upper bounds on APE for each theta in the approximated identified set
        APE_set_min_approx =[] #collects all lower bounds on APE for each theta in the approximated identified set

        for theta in theta_grid

        m = Model(Clp.Optimizer) #set up optimization and sets solver
        set_optimizer_attribute(m,"SolveType",1)
        set_silent(m) #prevent solver from showing all outputs

        #1) Declare variables

        #Indexing convention:
        #For x,y: 1st entry is "0", 2nd entry is "1"
        #For alpha: indexing follows alpha_grid

        #Order of dimensions: x2,y1,alpha,x1
        @variable(m, psi[1:2,1:2,1:K_approx,1:2])

        #2) Declare linear programming constraints
        constraint = Dict()

        #Constraint for psi to be positive
        constraint[1]=@constraint(m, psi[:,:,:,:] .>= 0)

        #Contraints for psi(;x1) to be a probabilitiy for x1=0,1
        constraint[2] = @constraint(m, 1-sum(psi[:,:,:,1]) == 0)
        constraint[3] = @constraint(m, 1-sum(psi[:,:,:,2]) == 0)

        counter=4
        #Constraints for psi to be consistent with the data
        for y2 in 0:1
                for y1 in 0:1
                        for x2 in 0:1
                                for x1 in 0:1
                                temp = map(alpha-> (softmax(theta*x2+alpha)^y2)*(1-softmax(theta*x2+alpha))^(1-y2),alpha_grid_approx)
                                constraint[counter]=@constraint(m, H[y2+1,x2+1,y1+1,x1+1]-temp'psi[x2+1,y1+1,:,x1+1]==0)
                                counter+=1
                                end
                        end
                end
        end

        #Constraints for psi to be consistent with outcome distribution in period 1
        for x1 in 0:1
                for y1 in 0:1
                        for (index,alpha) in enumerate(alpha_grid_approx)
                                temp = (softmax(theta*x1+alpha)^y1)*(1-softmax(theta*x1+alpha))^(1-y1)
                                constraint[counter]=@constraint(m,sum(psi[:,y1+1,index,x1+1])-temp*sum(psi[:,:,index,x1+1])==0)
                                counter+=1
                        end
                end
        end

        #3) Declare linear programming objective
        @objective(m, Min,0.5*sum(psi[:,:,:,1])+0.5*sum(psi[:,:,:,2]))

        #4) Solving the optimization problem
        JuMP.optimize!(m)

        #5) Add theta to the identified set if the problem is feasible
        # i.e if the constraints of the identified set are verified
        # and add the associated bounds for average partial effects
        if string(termination_status(m))=="OPTIMAL"
                push!(theta_set_approx,theta)

                #Upper bound for average partial effects associated to theta
                @objective(m,Max,sum(map(k->0.5*sum(psi[:,:,k,:])*(softmax(theta+alpha_grid_approx[k])-softmax(alpha_grid_approx[k])),collect(1:K_approx))))
                JuMP.optimize!(m)
                push!(APE_set_max_approx,objective_value(m))

                #Lower bound for average partial effects associated to theta
                @objective(m,Min,sum(map(k->0.5*sum(psi[:,:,k,:])*(softmax(theta+alpha_grid_approx[k])-softmax(alpha_grid_approx[k])),collect(1:K_approx))))
                JuMP.optimize!(m)
                push!(APE_set_min_approx,objective_value(m))

        end

        end

        #Report length of the identified set
        theta_set_max_approx = maximum(theta_set_approx)
        theta_set_min_approx = minimum(theta_set_approx)

        results[index,Symbol(string("theta_set_max_approx",K_approx))]=theta_set_max_approx
        results[index,Symbol(string("theta_set_min_approx",K_approx))]=theta_set_min_approx

        #Report global bounds on average partial effects
        APE_set_max_upper_bound = maximum(APE_set_max_approx)
        APE_set_min_lower_bound = minimum(APE_set_min_approx)

        results[index,Symbol(string("APE_set_max_upper_bound_approx",K_approx))]=APE_set_max_upper_bound
        results[index,Symbol(string("APE_set_min_lower_bound_approx",K_approx))]=APE_set_min_lower_bound
        end
end

plot_identified_set_theta_true_K31_Kapprox5 = plot(results,
Coord.Cartesian(ymin=-1.5,ymax=1.5),
layer(x=:theta_true,y=:theta_set_max,Geom.line,color=[colorant"grey0"]),
layer(x=:theta_true,y=:theta_set_min,Geom.line,color=[colorant"grey0"]),
layer(x=:theta_true,y=:theta_set_max_approx5,Geom.line,linestyle=[:dash],color=[colorant"grey49"]),
layer(x=:theta_true,y=:theta_set_min_approx5,Geom.line,linestyle=[:dash],color=[colorant"grey49"]),
Guide.xlabel("True parameter"),
Guide.ylabel("Identified region"),
Theme(panel_fill="white", grid_color="grey91",panel_stroke="grey15"))

draw(PDF(string(path_figures,"Identified_Set_Logit_T2_Predetermined_K31_Kapprox5.pdf"), 6inch, 4inch), plot_identified_set_theta_true_K31_Kapprox5)

plot_identified_set_theta_true_K31_Kapprox50 = plot(results,
Coord.Cartesian(ymin=-1.5,ymax=1.5),
layer(x=:theta_true,y=:theta_set_max,Geom.line,color=[colorant"grey0"]),
layer(x=:theta_true,y=:theta_set_min,Geom.line,color=[colorant"grey0"]),
layer(x=:theta_true,y=:theta_set_max_approx50,Geom.line,linestyle=[:dash],color=[colorant"grey49"]),
layer(x=:theta_true,y=:theta_set_min_approx50,Geom.line,linestyle=[:dash],color=[colorant"grey49"]),
Guide.xlabel("True parameter"),
Guide.ylabel("Identified region"),
Theme(panel_fill="white", grid_color="grey91",panel_stroke="grey15"))

draw(PDF(string(path_figures,"Identified_Set_Logit_T2_Predetermined_K31_Kapprox50.pdf"), 6inch, 4inch), plot_identified_set_theta_true_K31_Kapprox50)

plot_identified_set_theta_true_K31_Kapprox500 = plot(results,
Coord.Cartesian(ymin=-1.5,ymax=1.5),
layer(x=:theta_true,y=:theta_set_max,Geom.line,color=[colorant"grey0"]),
layer(x=:theta_true,y=:theta_set_min,Geom.line,color=[colorant"grey0"]),
layer(x=:theta_true,y=:theta_set_max_approx500,Geom.line,linestyle=[:dash],color=[colorant"grey49"]),
layer(x=:theta_true,y=:theta_set_min_approx500,Geom.line,linestyle=[:dash],color=[colorant"grey49"]),
Guide.xlabel("True parameter"),
Guide.ylabel("Identified region"),
Theme(panel_fill="white", grid_color="grey91",panel_stroke="grey15"))

draw(PDF(string(path_figures,"Identified_Set_Logit_T2_Predetermined_K31_Kapprox500.pdf"), 6inch, 4inch), plot_identified_set_theta_true_K31_Kapprox500)

plot_identified_set_APE_true_K31_Kapprox5 = plot(results,
Coord.Cartesian(ymin=-0.25,ymax=0.25),
Guide.yticks(ticks=collect(-0.25:0.1:0.25)),
layer(x=:APE_true,y=:APE_set_max_upper_bound,Geom.line,color=[colorant"grey0"]),
layer(x=:APE_true,y=:APE_set_min_lower_bound,Geom.line,color=[colorant"grey0"]),
layer(x=:APE_true,y=:APE_set_max_upper_bound_approx5,Geom.line,linestyle=[:dash],color=[colorant"grey49"]),
layer(x=:APE_true,y=:APE_set_min_lower_bound_approx5,Geom.line,linestyle=[:dash],color=[colorant"grey49"]),
Guide.xlabel("True APE"),
Guide.ylabel("Identified region"),
Theme(panel_fill="white", grid_color="grey91",panel_stroke="grey15"))

draw(PDF(string(path_figures,"Identified_Set_APE_Logit_T2_Predetermined_K31_Kapprox5.pdf"), 6inch, 4inch), plot_identified_set_APE_true_K31_Kapprox5)

plot_identified_set_APE_true_K31_Kapprox50 = plot(results,
Coord.Cartesian(ymin=-0.25,ymax=0.25),
Guide.yticks(ticks=collect(-0.25:0.1:0.25)),
layer(x=:APE_true,y=:APE_set_max_upper_bound,Geom.line,color=[colorant"grey0"]),
layer(x=:APE_true,y=:APE_set_min_lower_bound,Geom.line,color=[colorant"grey0"]),
layer(x=:APE_true,y=:APE_set_max_upper_bound_approx50,Geom.line,linestyle=[:dash],color=[colorant"grey49"]),
layer(x=:APE_true,y=:APE_set_min_lower_bound_approx50,Geom.line,linestyle=[:dash],color=[colorant"grey49"]),
Guide.xlabel("True APE"),
Guide.ylabel("Identified region"),
Theme(panel_fill="white", grid_color="grey91",panel_stroke="grey15"))

draw(PDF(string(path_figures,"Identified_Set_APE_Logit_T2_Predetermined_K31_Kapprox50.pdf"), 6inch, 4inch), plot_identified_set_APE_true_K31_Kapprox50)

plot_identified_set_APE_true_K31_Kapprox500 = plot(results,
Coord.Cartesian(ymin=-0.25,ymax=0.25),
Guide.yticks(ticks=collect(-0.25:0.1:0.25)),
layer(x=:APE_true,y=:APE_set_max_upper_bound,Geom.line,color=[colorant"grey0"]),
layer(x=:APE_true,y=:APE_set_min_lower_bound,Geom.line,color=[colorant"grey0"]),
layer(x=:APE_true,y=:APE_set_max_upper_bound_approx500,Geom.line,linestyle=[:dash],color=[colorant"grey49"]),
layer(x=:APE_true,y=:APE_set_min_lower_bound_approx500,Geom.line,linestyle=[:dash],color=[colorant"grey49"]),
Guide.xlabel("True APE"),
Guide.ylabel("Identified region"),
Theme(panel_fill="white", grid_color="grey91",panel_stroke="grey15"))

draw(PDF(string(path_figures,"Identified_Set_APE_Logit_T2_Predetermined_K31_Kapprox500.pdf"), 6inch, 4inch), plot_identified_set_APE_true_K31_Kapprox500)
