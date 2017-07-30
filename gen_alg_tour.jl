using DataFrames

#=
$ 0.01 per km

Agencies:
A (frequent flyer) = 3 times in a row --> 15% off the price of the third flight 
B (long distance) = more than 200 km --> 15% off the price of that flight 
C = if B was used in the previous flight, 20% off of this flight
D = every 10K km with this agency, $15 off (no percentage) the price
=#

cityPairs = readtable("sol64920.txt", separator=',', header=false)
cities = readtable("problem.csv", separator=',', header=false)
rename!(cities, Dict(:x1 => :city_id, :x2 => :xcoord, :x3 => :ycoord))
#cityPairs = cityPairs[:, [:x1, :x3]]
#rename!(cityPairs, Dict(:x1 => :origin, :x3 => :dest))
rename!(cityPairs, Dict(:x1 => :origin, :x2 => :agency, :x3 => :dest))

nCities = size(cities, 1)

nAgencies = 4
CORRECTION = 1e-5

abstract type CrossOverMethod end
immutable OnePoint <: CrossOverMethod end
immutable Uniform <: CrossOverMethod end

distToPrice(distance::Float64) = distance * 0.01
euclideanDist(x::Vector{Float64}, y::Vector{Float64}) = sqrt(sum((x - y).^2))

function getFitness(pop::Matrix{Int}, distances::Vector{Float64}, prices::Vector{Float64})::Vector{Float64}  
  costs = getCosts(pop, distances, prices)  
  #return 1./ (abs(vec(sum(costs, 2)) - sum(prices)) + CORRECTION)
  #return 1./(abs(log(vec(sum(costs, 2))) - log(sum(prices))) + CORRECTION)
  #return 1./(log(vec(sum(costs, 2))) + CORRECTION)
  #return 1./(log(sqrt(vec(sum(costs, 2)))) + CORRECTION)
  #fitness = sum(prices) - vec(sum(costs, 2))
  fitness = sum(prices) - vec(sum(costs, 2))
  fitness[fitness .< 0] = 0
  return fitness.^2
  #return fitness  
end

function getCosts(pop::Matrix{Int}, distances::Vector{Float64}, prices::Vector{Float64})::Matrix{Float64}
  function fA(i::Int, trip::Int)
    timesInARow += 1
    if (timesInARow == 3)
      costs[i, trip] *= 0.65
      timesInARow = 0
    end
  end

  function fB(i::Int, trip::Int)
    timesInARow = 0
    if (distances[trip] > 200)
      costs[i, trip] *= 0.85
    end
  end

  function fC(i::Int, trip::Int)
    timesInARow = 0
    if (trip > 1 && pop[i, trip - 1] == 2)
      costs[i, trip] *= 0.80
    end
  end

  function fD(i::Int, trip::Int)
    timesInARow = 0
    accDist += distances[trip]
    nDiscounts = floor(Int, accDist / 1e4)
    accDist -= nDiscounts * 1e4
    costs[i, trip] -= nDiscounts * 15    
  end

  
  (M, nCities) = size(pop)    
  costs = repmat(prices', M, 1)  
  funcs = [fA, fB, fC, fD]
  timesInARow = 0
  accDist = 0
  @sync @parallel for i in 1:M
    timesInARow = 0
    accDist = 0
    for trip in 1:nCities
      funcs[pop[i,trip]](i, trip)
    end
  end
  
  return costs
end



#roulette selection method
function chooseIndividual(probs::Vector{Float64})  
  p = rand()
  accum = 0
  i = 1
  while (accum < p) && (i < length(probs))
    accum += probs[i]
    if accum < p
      i += 1
    end
  end
  return i
end

function mutate(example::Vector{Int}, pMut::Float64)
  for i in 1:length(example)
    if rand() <= pMut
      example[i] = rand(1:nAgencies)
    end
  end  
  return example
end

#one-point cross-over
function cross(parent1::Vector{Int}, parent2::Vector{Int}, pCross::Float64, pMut::Float64, crossOverMethod::OnePoint)
  N = length(parent1)
  cutPoint = floor(Int, rand() * N) + 1
  if rand() < pCross
    child1 = [parent1[1:cutPoint] ; parent2[cutPoint+1:end]]
    child2 = [parent2[1:cutPoint] ; parent1[cutPoint+1:end]]
  else
    child1 = parent1
    child2 = parent2
  end  
  child1 = mutate(child1, pMut)
  child2 = mutate(child2, pMut)
  return (child1, child2)
end

function cross(parent1::Vector{Int}, parent2::Vector{Int}, pCross::Float64, pMut::Float64, crossOverMethod::Uniform)
  N = length(parent1)
  genes = rand(N) .> pCross
  child1 = zeros(parent1)
  child2 = zeros(parent1)
  child1[genes] = parent1[genes]
  child1[.!genes] = parent2[.!genes]
  child2[genes] = parent2[genes]
  child2[.!genes] = parent1[.!genes]  
  child1 = mutate(child1, pMut)
  child2 = mutate(child2, pMut)
  return (child1, child2)
end

function makeNewPop(pop::Matrix{Int}, probs::Vector{Float64}, pCross::Float64, pMut::Float64, crossOverMethod::CrossOverMethod)
  newPop = zeros(pop)  
  M = size(pop, 1)
  @sync @parallel for i = 1:2:M-1    
    best1 = chooseIndividual(probs)
    best1Prob = probs[best1]
    probs[best1] = 0
    best2 = chooseIndividual(probs)
    probs[best1] = best1Prob
    (newPop[i, :], newPop[i+1, :]) = cross(pop[best1,:], pop[best2,:], pCross, pMut, crossOverMethod)   
  end
  return newPop
end

function keepBestIndividual!(best::Vector{Int}, bestFitness::Float64, newPop::Matrix{Int}, newPopFitness::Vector{Float64})
  toReplace = indmin(newPopFitness)
  newPop[toReplace, :] = best
  newPopFitness[toReplace] = bestFitness
end

function elitism(pop::Matrix{Int}, fitness::Vector{Float64}, newPop::Matrix{Int}, newPopFitness::Vector{Float64})
  M = size(pop, 1)
  fullpop = [pop ; newPop]
  indices = sortperm([fitness ; newPopFitness], rev=true)
  return fullpop[indices[1:M],:]
end

popSize = 30
maxGen = 200
pCross = 0.8
pMut = 0.01
USE_ELITISM = true
#crossOverMethod = OnePoint()
crossOverMethod = Uniform()

mapping = Dict("A" => 1, "B" => 2, "C" => 3, "D" => 4)
originalSolution = convert(Matrix{Int}, [mapping[c] for c in cityPairs[:agency]]')

pop = repmat(originalSolution, popSize, 1)
pop = makeNewPop(pop, ones(popSize), 0.0, 0.5, crossOverMethod)
println("setup")
@time begin
distances = map(1:nCities) do i
      (originId, destId) = (cityPairs[i, :origin], cityPairs[i, :dest])      
      originCoordsDF = cities[cities[:city_id] .== originId, [:xcoord, :ycoord]]
      destCoordsDF = cities[cities[:city_id] .== destId, [:xcoord, :ycoord]]
      originCoords = vec(convert(Matrix{Float64}, originCoordsDF))
      destCoords = vec(convert(Matrix{Float64}, destCoordsDF))
      euclideanDist(originCoords, destCoords)
    end 
prices = map(distToPrice, distances)    
costsToOptimize = vec(getCosts(originalSolution, distances, prices))
end

gen = 1
champions = zeros(Int, maxGen, nCities)
championsFitness = zeros(maxGen)
fitness = getFitness(pop, distances, prices)
while gen <= maxGen  
  println("gen = $(gen) maxFitness=$(maximum(fitness)) minFitness=$(minimum(fitness)) avgFitness=$(mean(fitness))")
  best = indmax(fitness)
  champions[gen, :] = pop[best, :]
  championsFitness[gen] = fitness[best]
  probs = fitness ./ sum(fitness)
  newPop = makeNewPop(pop, probs, pCross, pMut, crossOverMethod)
  newPopFitness = getFitness(newPop, distances, prices)
  if USE_ELITISM
    keepBestIndividual!(pop[best, :], fitness[best], newPop, newPopFitness)
    #newPop = elitism(pop, fitness, newPop, newPopFitness)
  end
  pop = newPop
  fitness = newPopFitness
  gen += 1  
end
best = indmax(fitness)
#println("best solution of last pop = $(pop[best,:])")
println("best fitness = $(fitness[best])")
bestChampion = indmax(championsFitness)
#println("best solution = $(champions[bestChampion,:]), gen = $(bestChampion)")
println("best champion fitness = $(championsFitness[bestChampion])")
cost = getCosts(champions[bestChampion:bestChampion,:], distances, prices)
println("final cost = $(sum(cost))")
#println("original cost = $(sum(prices))")
println("original cost = $(sum(costsToOptimize))")

