import numpy as np
from distributions import s1_distribution, s2_distribution, systolic_distribution, diastolic_distribution


def hsmm_viterbi(posteriors, max_duration, centroid_dict, points_array, systolic_data):
    T,N = posteriors.shape
    A = [[0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1],
     [1, 0, 0, 0]]

    
    psi_np = np.empty((T+max_duration, N), dtype= np.float32)
    psi_np[:] = -np.inf
    psi_np[0, :] = np.log(posteriors[0, :])
    
    psi_arg_np = np.empty((T+max_duration, N), dtype= np.intc)
    psi_duration_np = np.empty((T+max_duration, N), dtype= np.intc)
    
    delta = psi_np
    psi = psi_arg_np
    psi_duration = psi_duration_np
    
    for t in range(0, T+max_duration):
        for s in range(N):
            for d in range(1, max_duration+1):
                start_t = max(0, min(t-d, T-1))
                end_t = min(t, T)
                
                delta_max = -np.inf
                i_max = -1
                for i in range(N):
                    temp_delta = delta[start_t, i] + np.log(A[i][s] + np.finfo(float).tiny)
                    if temp_delta > delta_max:
                        delta_max = temp_delta
                        i_max = i
                
                product_observation_probs = 0
                for i in range(start_t, end_t):
                    product_observation_probs+=np.log(posteriors[i,s] + np.finfo(float).tiny)
                    
                delta_this_duration = delta_max + product_observation_probs 
                # delta_this_duration = delta_max + product_observation_probs + np.log(durations[d-1, s] + np.finfo(float).tiny)
                duration_ms = (d-1)*20
                if s == 0:
                    delta_this_duration+=s1_distribution(x= duration_ms)
                elif s == 1:
                    delta_this_duration+=systolic_distribution(x= duration_ms, systolic_data= systolic_data)
                elif s == 2:
                    delta_this_duration+=s2_distribution(x= duration_ms)
                else:
                    delta_this_duration+=diastolic_distribution(x= duration_ms, centroid_dict= centroid_dict, points_array= points_array, systolic_data= systolic_data)
                if delta_this_duration > delta[t,s]:
                    delta[t,s] = delta_this_duration
                    psi[t,s] = i_max
                    psi_duration[t,s] = d
    
    current_state = -1
    end_time = -1
    max_delta_after = -np.inf
    
    for t in range(T, T+max_duration):
        for s in range(N):
            if delta[t,s] > max_delta_after:
                current_state = s
                end_time = t
                max_delta_after = delta[t,s]
                
    states_np = -np.ones(T+max_duration, dtype= np.int32)
    states = states_np
    states[end_time] = current_state
    t = end_time
    
    while t > 0:
        d = psi_duration[t, current_state]
        for i in range(max(0, t - d), t):
            states[i] = current_state

        t = max(0, t - d)

        current_state = psi[t, current_state]

    return np.array(states_np[:T])