########################################################################
# Goal: computation of the identified set for the probit case with T=2
# Julia Version 1.5.3
########################################################################

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

#Path to save figures (to specify depending on the user)
path_figures =""


########################################################################
# Functions
########################################################################

#Returns probit CDF at u
function cdfnormal(u)
        return cdf(Normal(0,1),u)
end

#Computes probability of a choice history given alpha and x1
function true_conditional_probability_given_alpha_x1(theta,y2,x2,y1,x1,alpha)
        #covariates X are iid Bernouilli(1/2) independent of everything else
        H_alpha = (cdfnormal(theta*x2+alpha)^y2)*((1-cdfnormal(theta*x2+alpha))^(1-y2))*(1/2)*(cdfnormal(theta*x1+alpha)^y1)*((1-cdfnormal(theta*x1+alpha))^(1-y1))
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
                 APE+=(cdfnormal(theta+alpha)-cdfnormal(alpha))*pi_alpha[index]
        end

        return APE
end

########################################################################
# Computation of the identified set for the predetermined case
# and the strictly exogenous case
########################################################################

#Model primitives
T=2  #Time periods
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
theta_set_max_SE =zeros(N_theta),theta_set_min_SE = zeros(N_theta),
APE_true = APE_true_grid,
APE_set_max_upper_bound = zeros(N_theta), APE_set_min_lower_bound =  zeros(N_theta),
APE_set_max_upper_bound_SE = zeros(N_theta), APE_set_min_lower_bound_SE =  zeros(N_theta)
)

#### Predetermined case
for (index,theta_true) in enumerate(theta_true_grid)
        println("Theta true : ",theta_true)

        #Vector of true probabilities
        H = true_probabilities(theta_true,alpha_grid,pi_alpha)

        #Identified set for predetermined case
        theta_set = []
        APE_set_max = [] #collects all upper bounds on APE for each theta in the identified set
        APE_set_min =[] #collects all lower bounds on APE for each theta in the identified set

        for theta in theta_grid
                m = Model(Clp.Optimizer) #set up optimization and sets solver
                set_optimizer_attribute(m,"SolveType",1)
                set_time_limit_sec(m, 60.0) # maximum search time of 60s
                set_silent(m) #prevent solver from showing all outputs

                #1) Declare variables

                #Indexing convention:
                #For x,y: 1st entry is "0", 2nd entry is "1"
                #For alpha: indexing follows alpha_grid

                #Order of dimensions: x2,y1,alpha,x1
                @variable(m, psi[1:2,1:2,1:K,1:2])

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
                                        temp = map(alpha-> (cdfnormal(theta*x2+alpha)^y2)*(1-cdfnormal(theta*x2+alpha))^(1-y2),alpha_grid)
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
                                        temp = (cdfnormal(theta*x1+alpha)^y1)*(1-cdfnormal(theta*x1+alpha))^(1-y1)
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
                        @objective(m,Max,sum(map(k->0.5*sum(psi[:,:,k,:])*(cdfnormal(theta+alpha_grid[k])-cdfnormal(alpha_grid[k])),collect(1:K))))
                        JuMP.optimize!(m)
                        push!(APE_set_max,objective_value(m))

                        #Lower bound for average partial effects associated to theta
                        @objective(m,Min,sum(map(k->0.5*sum(psi[:,:,k,:])*(cdfnormal(theta+alpha_grid[k])-cdfnormal(alpha_grid[k])),collect(1:K))))
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

#### Strictly exogenous case
for (index,theta_true) in enumerate(theta_true_grid)
        println("Theta true : ",theta_true)

        #Vector of true probabilities.
        H = true_probabilities(theta_true,alpha_grid,pi_alpha)

        #Identified set for strictly exogenous case
        theta_set_SE = []
        APE_set_max_SE = [] #collects all upper bounds on APE for each theta in the identified set
        APE_set_min_SE =[] #collects all lower bounds on APE for each theta in the identified set

        for theta in theta_grid
                m_SE = Model(Clp.Optimizer) #set up optimization and sets solver
                set_optimizer_attribute(m_SE,"SolveType",1)
                set_time_limit_sec(m_SE, 60.0) # maximum search time of 60s
                set_silent(m_SE) #prevent solver from showing all outputs

                #1) Declare variables

                #Indexing convention:
                #For x,y: 1st entry is "0", 2nd entry is "1"
                #For alpha: indexing follows alpha_grid

                #Order of dimensions: x2,y1,alpha,x1
                @variable(m_SE, psi[1:2,1:2,1:K,1:2])

                #2) Declare linear programming constraints
                constraint_SE = Dict()

                #Constraint for psi to be positive
                constraint_SE[1]=@constraint(m_SE, psi[:,:,:,:] .>= 0)

                #Contraints for psi(;x1) to be a probabilitiy for x1=0,1
                constraint_SE[2] = @constraint(m_SE, 1-sum(psi[:,:,:,1]) == 0)
                constraint_SE[3] = @constraint(m_SE, 1-sum(psi[:,:,:,2]) == 0)

                counter=4
                #Constraints for psi to be consistent with the data
                for y2 in 0:1
                        for y1 in 0:1
                                for x2 in 0:1
                                        for x1 in 0:1
                                        temp = map(alpha-> (cdfnormal(theta*x2+alpha)^y2)*(1-cdfnormal(theta*x2+alpha))^(1-y2),alpha_grid)
                                        constraint_SE[counter]=@constraint(m_SE, H[y2+1,x2+1,y1+1,x1+1]-temp'psi[x2+1,y1+1,:,x1+1]==0)
                                        counter+=1
                                        end
                                end
                        end
                end

                #Constraints for psi to be consistent with outcome distribution in period 1
                for x1 in 0:1
                        for y1 in 0:1
                                for (index,alpha) in enumerate(alpha_grid)
                                        temp = (cdfnormal(theta*x1+alpha)^y1)*(1-cdfnormal(theta*x1+alpha))^(1-y1)
                                        constraint_SE[counter]=@constraint(m_SE,sum(psi[:,y1+1,index,x1+1])-temp*sum(psi[:,:,index,x1+1])==0)
                                        counter+=1
                                end
                        end
                end

                #Constraints for psi under strict exogeneity
                for x2 in 0:1
                        for x1 in 0:1
                                for (index,alpha) in enumerate(alpha_grid)
                                        constraint_SE[counter]=@constraint(m_SE,psi[x2+1,2,index,x1+1]*(1-cdfnormal(theta*x1+alpha)) - psi[x2+1,1,index,x1+1]*cdfnormal(theta*x1+alpha)== 0)
                                        counter+=1
                                end
                        end
                end

                #3) Declare linear programming objective
                @objective(m_SE, Min,sum(psi))

                #4) Solving the optimization problem
                JuMP.optimize!(m_SE)

                #5) Add theta to the identified set if the problem is feasible
                # i.e if the constraints of the identified set are verified
                # and add the associated bounds for average partial effects
                if string(termination_status(m_SE))=="OPTIMAL"
                        push!(theta_set_SE,theta)

                        #Upper bound for average partial effects associated to theta
                        @objective(m_SE,Max,sum(map(k->0.5*sum(psi[:,:,k,:])*(cdfnormal(theta+alpha_grid[k])-cdfnormal(alpha_grid[k])),collect(1:K))))
                        JuMP.optimize!(m_SE)
                        push!(APE_set_max_SE,objective_value(m_SE))

                        #Lower bound for average partial effects associated to theta
                        @objective(m_SE,Min,sum(map(k->0.5*sum(psi[:,:,k,:])*(cdfnormal(theta+alpha_grid[k])-cdfnormal(alpha_grid[k])),collect(1:K))))
                        JuMP.optimize!(m_SE)
                        push!(APE_set_min_SE,objective_value(m_SE))

                end
        end

        #Report length of the identified set
        theta_set_max_SE = maximum(theta_set_SE)
        theta_set_min_SE = minimum(theta_set_SE)

        results[index,:theta_set_max_SE]=theta_set_max_SE
        results[index,:theta_set_min_SE]=theta_set_min_SE

        #Report global bounds on average partial effects
        APE_set_max_upper_bound_SE = maximum(APE_set_max_SE)
        APE_set_min_lower_bound_SE = minimum(APE_set_min_SE)

        results[index,:APE_set_max_upper_bound_SE]=APE_set_max_upper_bound_SE
        results[index,:APE_set_min_lower_bound_SE]=APE_set_min_lower_bound_SE

end

plot_identified_set_APE_true = plot(results,
Coord.Cartesian(ymin=-0.35,ymax=0.35,xmin=-0.27,xmax=0.27),
Guide.yticks(ticks=collect(-0.35:0.1:0.35)),
layer(x=:APE_true,y=:APE_set_max_upper_bound,Geom.line,color=[colorant"grey0"]),
layer(x=:APE_true,y=:APE_set_min_lower_bound,Geom.line,color=[colorant"grey0"]),
layer(x=:APE_true,y=:APE_set_max_upper_bound_SE,Geom.line,linestyle=[:dash],color=[colorant"grey49"]),
layer(x=:APE_true,y=:APE_set_min_lower_bound_SE,Geom.line,linestyle=[:dash],color=[colorant"grey49"]),
Guide.xlabel("True APE"),
Guide.ylabel("Identified region"),
Theme(panel_fill="white", grid_color="grey91",panel_stroke="grey15"))

draw(PDF(string(path_figures,"Identified_Set_APE_Probit_T2.pdf"), 6inch, 4inch), plot_identified_set_APE_true)

plot_identified_set_theta_true = plot(results,
Coord.Cartesian(ymin=-1.5,ymax=1.5),
layer(x=:theta_true,y=:theta_set_max,Geom.line,color=[colorant"grey0"]),
layer(x=:theta_true,y=:theta_set_min,Geom.line,color=[colorant"grey0"]),
layer(x=:theta_true,y=:theta_set_max_SE,Geom.line,linestyle=[:dash],color=[colorant"grey49"]),
layer(x=:theta_true,y=:theta_set_min_SE,Geom.line,linestyle=[:dash],color=[colorant"grey49"]),
Guide.xlabel("True parameter"),
Guide.ylabel("Identified region"),
Theme(panel_fill="white", grid_color="grey91",panel_stroke="grey15"))

draw(PDF(string(path_figures,"Identified_Set_Probit_T2.pdf"), 6inch, 4inch), plot_identified_set_theta_true)
