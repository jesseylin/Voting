using TensorOperations
using LinearAlgebra

function ⊗(a, b)
	rank_a = ndims(a)
	rank_b = ndims(b)
	indices_a = Vector(1:rank_a)
	indices_b = Vector(1:rank_b) .+ rank_a

	ncon([a,b], [-indices_a, -indices_b])
end

# VotePreference object
struct VotePreference
	prefvecs
	prefstate
	num_candidates
end
function VotePreference(prefvecs::Vector{Any})
	prefstate = prefvecs[1]
	for i in 2:length(prefvecs)
		prefstate = prefstate ⊗ prefvecs[i]
	end
	num_candidates = length(prefvecs[1])
	return VotePreference(prefvecs, prefstate, num_candidates)
end
function VotePreference(prefvecs::Array)
	prefstate = prefvecs[:,1]
	for i in 2:size(prefvecs)[2]
		prefstate = prefstate ⊗ prefvecs[:,i]
	end
	num_candidates = size(prefvecs)[1]
	return VotePreference(prefvecs, prefstate, num_candidates)
end
function VotePreference(;prefstate=0)
	num_candidates = size(prefstate)[1]
	return VotePreference(nothing, prefstate, num_candidates)
end
function ⋅(x::VotePreference, y::VotePreference)
	""" Computes the overlap of the tensor product states
	"""
	x = x.prefstate
	y = y.prefstate
	indices_x = [1:ndims(x)...]
	indices_y = [1:ndims(y)...]
	if indices_x != indices_y
		throw(ArgumentError("Arguments have preference states with unequal dimensionality."))
	end
	ncon([x,y], [indices_x, indices_y])
end

begin
	num_voters = 10
	num_offices = 3
	num_candidates = 3
end

begin
	voters = []
	for i in 1:num_voters
		offices = []
		for j in 1:num_offices
			v = rand(num_candidates)
			v /= norm(v)
			push!(offices, v)
		end
		voter = VotePreference(offices)
		push!(voters, voter)
	end
end

# FPP basics
begin
	function votes_for_office(voters, office_idx)
		num_candidates = voters[1].num_candidates
		vote_count = zeros(num_candidates)
		for voter in voters
			vote_getter = argmax(voter.prefvecs[office_idx])
			vote_count[vote_getter] += 1
		end
		return vote_count
	end
    function fpp(voters, num_offices, num_candidates)
        ballot = []
        for i in 1:num_offices
            push!(ballot, votes_for_office(voters, i))
        end

        outcomes = []
        winners = Set([])
        for i in 1:num_offices
            v = zeros(Int, num_candidates)
            officeballot = ballot[i]
            winner_idx = argmax(officeballot)
            while winner_idx in winners
                # set the winner_idx to -1 so it does not get counted again
                officeballot[winner_idx] = -1
                winner_idx = argmax(officeballot)
            end
            push!(winners, winner_idx)
            v[winner_idx] = 1
            push!(outcomes, v)
        end
        outstate = VotePreference(outcomes)
    end
    function get_utility(voters, outstate)
        utility = 0
        for voter in voters
            utility = utility + log(only(voter ⋅ outstate))
        end
        return utility
    end
    function find_optimal(voters)
        num_candidates = voters[1].num_candidates
        all_outcomes = similar(voters[1].prefstate, Any)
        all_utilities = similar(all_outcomes, AbstractFloat)
        dimension = size(all_outcomes)
        tmp = I(length(all_outcomes))
        
        for i in 1:length(all_outcomes)
            tmpvec = Vector{Int}(tmp[:,i])
            all_outcomes[i] = VotePreference(
                prefstate=reshape(tmpvec, dimension...)
            )
        end

        for (i, o) in enumerate(all_outcomes)
            all_utilities[i] = get_utility(voters, o)
        end
        return findmax(all_utilities)
    end
end