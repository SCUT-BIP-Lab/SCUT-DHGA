'''
@Description: Calculate equal error rate
@Date: 2019-11-27 21:29:36
@LastEditors: lxy
@LastEditTime: 2019-11-28 10:07:10
'''
import numpy as np




def calculate_eer(features1, features2, pair_labels, useCosin):
    assert(features1.shape == features2.shape)

    # linear
    if(len(features1.shape) <= 2):
        if(useCosin):
            # cosin distances
            distances = 1 - np.sum(features1 * features2, axis=1) / (np.linalg.norm(features1, axis=1) * np.linalg.norm(features2, axis=1))
        else:
            # euclidean distances
            distances = np.sqrt(np.sum((features1 - features2) ** 2, axis=1))

    # time distributed
    else:
        if(useCosin):
            # cosin distances
            distances = np.mean(1 - np.sum(features1 * features2, axis=2) / (np.linalg.norm(features1, axis=2) * np.linalg.norm(features2, axis=2)), axis=1)
        else:
            # euclidean distances
            distances = np.mean(np.sqrt(np.sum((features1 - features2) ** 2, axis=2)), axis=1)

    min_dis = np.min(distances)
    max_dis = np.max(distances)

    accept_distances = distances[pair_labels == True]
    reject_distances = distances[pair_labels == False]

    FARs = []
    FRRs = []
    thresholds = []
    errors = []
    for threshold in np.linspace(min_dis, max_dis, num=10000): # threshold linspace
        thresholds.append(threshold)
        FRR = np.sum(accept_distances >= threshold) / accept_distances.shape[0]
        FAR = np.sum(reject_distances < threshold) / reject_distances.shape[0]
        FRRs.append(FRR)
        FARs.append(FAR)
        errors.append(abs(FAR - FRR))
    min_errors_idx = np.argmin(np.asarray(errors))
    EER = (FRRs[min_errors_idx] + FARs[min_errors_idx]) / 2
    best_thres = thresholds[min_errors_idx]

    return EER, best_thres