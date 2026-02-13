#include <iostream>
#include <cctype>
#include <limits>
#include <queue>
#include <stack>
#include <algorithm>
#include <cassert>
#include "fold.h"

//static
bool
Fold::
allow_paired(char x, char y)
{
    x = std::tolower(x);
    y = std::tolower(y);
    return (x=='a' && y=='u') || (x=='u' && y=='a') || 
        (x=='c' && y=='g') || (x=='g' && y=='c') ||
        (x=='g' && y=='u') || (x=='u' && y=='g');
}

//static
auto
Fold::
make_paren(const std::vector<uint32_t>& p) -> std::string
{
    std::string s(p.size()-1, '.');
    for (size_t i=1; i!=p.size(); ++i)
    {
        if (p[i] != 0)
            s[i-1] = p[i]>i ? '(' : ')';
    }
    return s;
}

auto
Fold::Options::
make_constraint(const std::string& seq, bool canonical_only /*=true*/) 
    -> std::pair<std::vector<std::vector<bool>>, std::vector<std::vector<bool>>>
{
    const auto L = seq.size();
    // Normalise per-position constraint spec to 1-based indexing (0th unused)
    constraint_spec.reserve(L+1);
    while (constraint_spec.size() <= L)
        constraint_spec.push_back(Options::ANY);
    constraint_spec.resize(L+1);

    // Drop disallowed enforced pairs (non-canonical or too-short hairpin)
    for (auto i=L; i>=1; i--)
        if (constraint_spec[i] > 0 && constraint_spec[i] <= L) // paired (since bases are 1-based indexed)
            if ( (canonical_only && !Fold::allow_paired(seq[i-1], seq[constraint_spec[i]-1])) || // delete non-canonical base-pairs
                    (constraint_spec[i] - i <= min_hairpin) ) // delete very short hairpin
                constraint_spec[i] = constraint_spec[constraint_spec[i]] = Options::UNPAIRED;

    // Flag positions involved in crossing (pseudoknotted) pairs
    std::vector<bool> is_in_pseudoknot(L+1, false);
    for (auto i=1; i<=L; i++)
        if (constraint_spec[i] > 0 && constraint_spec[i] <= L) // paired
            for (auto k=i+1; k<constraint_spec[i]; k++)
                if (/*constraint_spec[k] > 0 &&*/ constraint_spec[k] <= L && constraint_spec[k] > constraint_spec[i]) // paired & is_in_pseudoknot
                    is_in_pseudoknot[i] = is_in_pseudoknot[constraint_spec[i]] = is_in_pseudoknot[k] = is_in_pseudoknot[constraint_spec[k]] = true;

    // Count explicit pair candidates to detect positions that appear in exactly one pair
    std::vector<uint32_t> candidate_involvement_cnt(L+1, 0);
    for (auto p: constraint_pair_candidates) 
    {
        const auto [p1, p2] = std::minmax(p.first, p.second);
        if (p2-p1 > min_hairpin && (!canonical_only || Fold::allow_paired(seq[p1-1], seq[p2-1])))
        {
            candidate_involvement_cnt[p1]++;
            candidate_involvement_cnt[p2]++;
        }
    }

    std::vector<bool> just_once_paired(L+1, false);
    for (auto p: constraint_pair_candidates)
    {
        if (candidate_involvement_cnt[p.first]==1 && candidate_involvement_cnt[p.second]==1)
        {
            just_once_paired[p.first] = true;
            just_once_paired[p.second] = true;
        }
    }

    // Mark pseudoknots implied by explicit pair candidates
    for (auto p: constraint_pair_candidates)
    {
        const auto [p1, p2] = std::minmax(p.first, p.second);
        if (p2-p1 > min_hairpin && (!canonical_only || Fold::allow_paired(seq[p1-1], seq[p2-1])))
        {
            for (auto q: constraint_pair_candidates)
            {
                if (p==q) continue;
                const auto [q1, q2] = std::minmax(q.first, q.second);
                if (q2-q1 > min_hairpin && (!canonical_only || Fold::allow_paired(seq[q1-1], seq[q2-1])))
                    if (p1 < q1 && q1 < p2 && p2 < q2)
                        is_in_pseudoknot[p1] = is_in_pseudoknot[p2] = is_in_pseudoknot[q1] = is_in_pseudoknot[q2] = true;
            }
        }
    }

    // Build DP-usable allow tables from constraints
    std::vector<std::vector<bool>> allow_paired(L+1, std::vector<bool>(L+1, false));
    std::vector<std::vector<bool>> allow_unpaired(L+1, std::vector<bool>(L+1, false));
    for (auto i=L; i>=1; i--)
    {
        allow_unpaired[i][i-1] = true; // the empty string is alway allowed to be unpaired
        allow_unpaired[i][i] = constraint_spec[i]==Options::ANY || constraint_spec[i]==Options::UNPAIRED || is_in_pseudoknot[i];
        if (just_once_paired[i] && !is_in_pseudoknot[i])
            allow_unpaired[i][i] = false;
        bool bp_l = constraint_spec[i]==Options::ANY || constraint_spec[i]==Options::PAIRED_L || constraint_spec[i]==Options::PAIRED_LR;
        for (auto j=i+1; j<=L; j++)
        {
            allow_paired[i][j] = j-i > min_hairpin;
            bool bp_r = constraint_spec[j]==Options::ANY || constraint_spec[j]==Options::PAIRED_R || constraint_spec[j]==Options::PAIRED_LR;
            allow_paired[i][j] = allow_paired[i][j] && ((bp_l && bp_r) || constraint_spec[i]==j);
            if (canonical_only)
                allow_paired[i][j] = allow_paired[i][j] && Fold::allow_paired(seq[i-1], seq[j-1]);
            allow_unpaired[i][j] = allow_unpaired[i][j-1] && allow_unpaired[j][j];
        }
    }

    // Explicitly allow any pair in the candidate list that survives filtering
    for (auto p: constraint_pair_candidates)
    {
        const auto [p1, p2] = std::minmax(p.first, p.second);
        if (p2-p1 > min_hairpin && (!canonical_only || Fold::allow_paired(seq[p1-1], seq[p2-1])))
            allow_paired[p1][p2] = true;
    }

    return std::make_pair(allow_paired, allow_unpaired);
}

auto 
Fold::Options::
make_penalty(size_t L) 
    -> std::tuple<TriMatrix<float>, std::vector<std::vector<float>>, float>
{
    TriMatrix<float> p_paired(L+1, 0.0);
    std::vector<std::vector<float>> p_unpaired(L+1, std::vector<float>(L+1, 0.0));
    float p_const = 0;
    if (use_penalty)
    {
        if (ref2.size() == 0)
        {
            for (auto i=L; i>=1; i--)
            {
                if (ref[i]==Options::ANY || ref[i]==Options::UNPAIRED)
                {
                    p_unpaired[i][i] = -pos_unpaired;
                    p_const += pos_unpaired;
                }
                else
                    p_unpaired[i][i] = neg_unpaired;

                for (auto j=i+1; j<=L; j++)
                {
                    p_unpaired[i][j] = p_unpaired[i][j-1] + p_unpaired[j][j];

                    if (ref[i] == j)
                    {
                        p_paired[i][j] = -pos_paired;
                        p_const += pos_paired;
                    }
                    else
                        p_paired[i][j] = neg_paired;
                }
            }
        }            
        else
        {
            std::vector<bool> paired(L+1, false);
            for (auto p: ref2)
                paired[p.first] = paired[p.second] = true;

            for (auto i=L; i>=1; i--)
            {
                if (!paired[i])
                {
                    p_unpaired[i][i] = -pos_unpaired;
                    p_const += pos_unpaired;
                }
                else
                    p_unpaired[i][i] = neg_unpaired;
                
                for (auto j=i+1; j<=L; j++)
                {
                    p_unpaired[i][j] = p_unpaired[i][j-1] + p_unpaired[j][j];
                    p_paired[i][j] = neg_paired;
                }
            }
            for (auto p: ref2)
            {
                const auto [p1, p2] = std::minmax(p.first, p.second);
                p_paired[p1][p2] = -pos_paired;
                p_const += pos_paired;
            }
        }
    }
    return std::make_tuple(p_paired, p_unpaired, p_const);
}
