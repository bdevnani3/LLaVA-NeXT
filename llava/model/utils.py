from transformers import AutoConfig

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d
import heapq
import random
import torch

def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if "llava" in config and "llava" not in cfg.model_type:
        assert cfg.model_type == "llama"
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = "LlavaLlamaForCausalLM"
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)

##############################
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d
import heapq

task_to_clip_cache_path ={
    "egoschema_subset": "/nethome/bdevnani3/flash/.cache/results_final/clip_cache_similarity/egoschema_subset/",
    "longvideobench_val_v": "/nethome/bdevnani3/flash/.cache/results_final/clip_cache_similarity/longvideobench_val_v/",
    # TODO: add longvideobench_val_i
    "mlvu_dev": "/nethome/bdevnani3/flash/.cache/results_final/clip_cache_similarity/mlvu_dev/",
    "mvbench": "/nethome/bdevnani3/flash/.cache/results_final/clip_cache_similarity/",
    "nextqa_mc_test": "/nethome/bdevnani3/flash/.cache/results_final/clip_cache_similarity/nextqa_mc_test/",
    # TODO: add perceptiontest as it is in progress
    "tempcompass": "/nethome/bdevnani3/flash/.cache/results_final/clip_cache_similarity/",
    "videomme": "/nethome/bdevnani3/flash/.cache/results_final/clip_cache_similarity/videomme/",
}

num_frames_to_skip_value = {
    1600: 1,
    800: 2,
    400: 4,
    200: 8,
    100: 16,
    50: 32,
    25: 64,
}


def find_regions_of_interest(frame_similarities, text_similarities, 
                             window_size=15, prominence_threshold=0.1,
                             similarity_threshold=0.8, change_threshold=0.15):
    """
    Find regions of interest in a video based on frame similarities and text-image similarities.
    
    Parameters:
    -----------
    frame_similarities : array-like
        Array of similarities between consecutive frames (orange line)
    text_similarities : array-like
        Array of similarities between frames and query text (blue line)
    window_size : int
        Size of the sliding window for smoothing
    prominence_threshold : float
        Threshold for peak prominence in text similarities
    similarity_threshold : float
        Threshold below which frame similarity indicates a scene change
    change_threshold : float
        Threshold for significant changes in either signal
        
    Returns:
    --------
    dict
        Dictionary containing different types of regions of interest
    """
    # Ensure inputs are numpy arrays
    frame_similarities = np.array(frame_similarities)
    text_similarities = np.array(text_similarities)
    
    # 1. Apply smoothing to reduce noise
    frame_smooth = savgol_filter(frame_similarities, window_size, 3)
    text_smooth = savgol_filter(text_similarities, window_size, 3)
    
    # 2. Detect scene changes (significant drops in frame similarity)
    scene_change_mask = frame_smooth < similarity_threshold
    scene_changes = np.where(np.diff(scene_change_mask.astype(int)) > 0)[0]
    
    # 3. Find text similarity peaks (frames highly relevant to the query)
    peaks, peak_props = find_peaks(text_smooth, prominence=prominence_threshold)
    
    # 4. Compute gradients to detect sudden changes
    text_gradient = np.abs(np.gradient(text_smooth))
    frame_gradient = np.abs(np.gradient(frame_smooth))
    
    # Normalize gradients to [0,1] range
    scaler = MinMaxScaler()
    text_gradient_norm = scaler.fit_transform(text_gradient.reshape(-1, 1)).flatten()
    frame_gradient_norm = scaler.fit_transform(frame_gradient.reshape(-1, 1)).flatten()
    
    # 5. Find regions with significant changes in either signal
    significant_changes = np.where((text_gradient_norm > change_threshold) | 
                                  (frame_gradient_norm > change_threshold))[0]
    
    # 6. Create a composite score
    # Weight: higher text similarity and stable frame similarity (no major changes)
    composite_score = text_smooth * (1 - np.abs(np.gradient(frame_smooth)))
    
    # Smooth the composite score
    composite_score = gaussian_filter1d(composite_score, sigma=window_size/3)
    
    # Find peaks in composite score
    composite_peaks, _ = find_peaks(composite_score, prominence=prominence_threshold)
    
    # 7. Segment the video based on scene changes
    segments = []
    if len(scene_changes) > 0:
        # Add start of video to beginning of scene changes
        all_changes = np.concatenate(([0], scene_changes, [len(frame_similarities) - 1]))
        
        for i in range(len(all_changes) - 1):
            start = all_changes[i]
            end = all_changes[i + 1]
            
            # Calculate mean text similarity for this segment
            mean_text_sim = np.mean(text_smooth[start:end])
            
            segments.append({
                'start': int(start),
                'end': int(end),
                'mean_text_similarity': float(mean_text_sim),
                'duration': int(end - start)
            })
    
    # 8. Get top segments by text similarity
    top_segments = sorted(segments, key=lambda x: x['mean_text_similarity'], reverse=True)
    
    # 9. Find regions with sustained high text similarity
    high_text_regions = []
    in_high_region = False
    start_idx = 0
    
    # Calculate dynamic threshold based on statistics
    text_mean = np.mean(text_smooth)
    text_std = np.std(text_smooth)
    high_text_threshold = text_mean + text_std
    
    for i, val in enumerate(text_smooth):
        if val > high_text_threshold and not in_high_region:
            # Start of new high region
            in_high_region = True
            start_idx = i
        elif val <= high_text_threshold and in_high_region:
            # End of high region
            in_high_region = False
            # Only include if region is substantial
            if i - start_idx > window_size / 2:
                high_text_regions.append({
                    'start': int(start_idx),
                    'end': int(i),
                    'duration': int(i - start_idx),
                    'mean_similarity': float(np.mean(text_smooth[start_idx:i]))
                })
    
    # If we ended in a high region, add it
    if in_high_region:
        high_text_regions.append({
            'start': int(start_idx),
            'end': int(len(text_smooth)),
            'duration': int(len(text_smooth) - start_idx),
            'mean_similarity': float(np.mean(text_smooth[start_idx:]))
        })
    
    # 10. Return all identified regions of interest
    regions_of_interest = {
        'scene_changes': scene_changes.tolist(),
        'text_similarity_peaks': peaks.tolist(),
        'significant_changes': significant_changes.tolist(),
        'composite_peaks': composite_peaks.tolist(),
        'segments': segments,
        'top_segments_by_text_similarity': top_segments[:3] if top_segments else [],
        'high_text_similarity_regions': high_text_regions
    }
    
    return regions_of_interest, composite_score

def extract_key_frames(frame_similarities, text_similarities, n=5, method='combined'):
    """
    Extract n key frames of high interest from the video.
    
    Parameters:
    -----------
    frame_similarities : array-like
        Array of similarities between consecutive frames
    text_similarities : array-like
        Array of similarities between frames and query text
    n : int
        Number of key frames to extract
    method : str
        Method to use for key frame extraction:
        - 'text': based purely on text similarity
        - 'change': based on frame changes
        - 'combined': using a composite score (default)
        - 'regions': frames from middle of high interest regions
        
    Returns:
    --------
    list
        Indices of key frames
    """
    regions, composite_score = find_regions_of_interest(
        frame_similarities, text_similarities)
    
    # Initialize frame scores based on the selected method
    if method == 'text':
        # Use text similarity directly
        frame_scores = savgol_filter(text_similarities, 15, 3)
    elif method == 'change':
        # Use inverse of frame similarity gradient (high = big change)
        gradient = np.abs(np.gradient(savgol_filter(frame_similarities, 15, 3)))
        frame_scores = gradient
    elif method == 'combined':
        # Use the composite score
        frame_scores = composite_score
    elif method == 'regions':
        # Extract frames from the middle of high interest regions
        key_frames = []
        
        # First, check high text similarity regions
        high_regions = sorted(
            regions['high_text_similarity_regions'], 
            key=lambda x: x['mean_similarity'], 
            reverse=True
        )
        
        # Add middle frame from each high text region
        for region in high_regions:
            mid_frame = (region['start'] + region['end']) // 2
            key_frames.append((mid_frame, region['mean_similarity']))
        
        # Then add middle frames from top segments
        for segment in regions['top_segments_by_text_similarity']:
            mid_frame = (segment['start'] + segment['end']) // 2
            # Only add if not too close to existing frames
            if all(abs(mid_frame - f[0]) > 30 for f in key_frames):
                key_frames.append((mid_frame, segment['mean_text_similarity']))
        
        # Finally, add text similarity peaks
        for peak in regions['text_similarity_peaks']:
            # Only add if not too close to existing frames
            if all(abs(peak - f[0]) > 30 for f in key_frames):
                key_frames.append((peak, text_similarities[peak]))
        
        # Sort by score and take top n
        key_frames.sort(key=lambda x: x[1], reverse=True)
        return [frame[0] for frame in key_frames[:n]]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # For methods other than 'regions', find local maxima and take top n
    if method != 'regions':
        # Find all peaks
        peaks, _ = find_peaks(frame_scores)
        
        # If not enough peaks, add highest non-peak values
        if len(peaks) < n:
            # Get indices of all frames sorted by score
            all_indices = np.argsort(frame_scores)[::-1]
            # Filter out existing peaks
            additional = [i for i in all_indices if i not in peaks]
            # Add as many as needed
            peaks = np.concatenate([peaks, additional[:n-len(peaks)]])
        
        # Score the peaks
        peak_scores = frame_scores[peaks]
        
        # Get top n peaks by score
        top_indices = np.argsort(peak_scores)[::-1][:n]
        key_frames = peaks[top_indices]
        
        return sorted(key_frames)

def detect_and_extract_distributed_keyframes(frame_similarities, text_similarities, 
                                            num_frames=5, min_subclip_duration=15,
                                            frame_threshold=0.8, text_weight=1):
    """
    Detects subclips in a video and extracts keyframes with balanced distribution across subclips.
    
    Parameters:
    -----------
    frame_similarities : array-like
        Array of similarities between consecutive frames
    text_similarities : array-like
        Array of similarities between frames and query text
    num_frames : int
        Total number of keyframes to extract
    min_subclip_duration : int
        Minimum number of frames to consider as a valid subclip
    frame_threshold : float
        Threshold below which frame similarity indicates a scene change
    text_weight : float
        Weight given to text similarity vs. frame changes (0-1)
        
    Returns:
    --------
    tuple
        (subclips, keyframes) where subclips is a list of dictionaries with subclip info
        and keyframes is a list of frame indices
    """
    # Get regions of interest and composite score
    regions, composite_score = find_regions_of_interest(
        frame_similarities, text_similarities, 
        similarity_threshold=frame_threshold)
    
    # 1. DETECT SUBCLIPS - Combine scene changes with text similarity analysis
    
    # Start with scene changes
    scene_changes = regions['scene_changes']
    
    # Add start and end frames if not already included
    if 0 not in scene_changes:
        scene_changes = np.concatenate(([0], scene_changes))
    if len(frame_similarities) - 1 not in scene_changes:
        scene_changes = np.concatenate((scene_changes, [len(frame_similarities) - 1]))
    
    # Sort scene changes
    scene_changes = np.sort(scene_changes)
    
    # Create initial subclips from scene changes
    subclips = []
    for i in range(len(scene_changes) - 1):
        start = int(scene_changes[i])
        end = int(scene_changes[i + 1])
        
        # Only include if duration meets minimum
        if end - start >= min_subclip_duration:
            # Calculate metrics for this subclip
            mean_text_sim = float(np.mean(text_similarities[start:end]))
            max_text_sim = float(np.max(text_similarities[start:end]))
            text_variance = float(np.var(text_similarities[start:end]))
            
            # Combined relevance score (weighted average of mean and max)
            # relevance_score = 0.7 * max_text_sim + 0.3 * mean_text_sim
            relevance_score = max_text_sim
            
            subclips.append({
                'start': start,
                'end': end,
                'duration': end - start,
                'mean_text_similarity': mean_text_sim,
                'max_text_similarity': max_text_sim,
                'text_variance': text_variance,
                'relevance_score': relevance_score
            })
    
    # If no valid subclips found, create one covering the entire video
    if not subclips:
        subclips = [{
            'start': 0,
            'end': len(frame_similarities) - 1,
            'duration': len(frame_similarities) - 1,
            'mean_text_similarity': float(np.mean(text_similarities)),
            'max_text_similarity': float(np.max(text_similarities)),
            'text_variance': float(np.var(text_similarities)),
            'relevance_score': float(np.mean(text_similarities))
        }]
    
    # Sort subclips by relevance score
    subclips.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # 2. EXTRACT DISTRIBUTED KEYFRAMES
    
    # Calculate how many frames to allocate to each subclip based on:
    # - Relevance score (more relevant subclips get more frames)
    # - Duration (longer subclips get more frames)
    total_score = sum(clip['relevance_score'] * np.sqrt(clip['duration']) for clip in subclips)
    
    if total_score == 0:  # Handle edge case
        total_score = 1
    
    # Initial frame allocation
    for clip in subclips:
        # Calculate proportion of frames based on relevance and duration
        proportion = (clip['relevance_score'] * np.sqrt(clip['duration'])) / total_score
        clip['allocated_frames'] = max(1, int(proportion * num_frames))
    
    # Adjust allocations to match requested total
    total_allocated = sum(clip['allocated_frames'] for clip in subclips)
    
    # Add or remove frames as needed
    if total_allocated < num_frames:
        # Distribute remaining frames to top subclips
        remaining = num_frames - total_allocated
        for i in range(min(remaining, len(subclips))):
            subclips[i]['allocated_frames'] += 1
    elif total_allocated > num_frames:
        # Remove frames from least relevant subclips
        excess = total_allocated - num_frames
        for i in range(1, min(excess + 1, len(subclips))):
            if subclips[-i]['allocated_frames'] > 1:  # Ensure at least one frame per clip
                subclips[-i]['allocated_frames'] -= 1
    
    # 3. EXTRACT FRAMES FROM EACH SUBCLIP
    keyframes = []
    
    for clip in subclips:
        frames_to_extract = clip['allocated_frames']
        start, end = clip['start'], clip['end']
        
        if frames_to_extract == 0:
            continue
        elif frames_to_extract == 1:
            # Just take the frame with highest text similarity
            best_frame = start + np.argmax(text_similarities[start:end])
            keyframes.append(int(best_frame))
        else:
            # Create a score for each frame in the subclip
            frame_scores = np.zeros(end - start)
            
            # Calculate scores based on text similarity and frame changes
            smoothed_text = savgol_filter(text_similarities[start:end], 
                                         min(15, (end-start)//2*2+1), 3)
            
            # For frame changes, we want peaks in the gradient (big changes)
            if start > 0 and end < len(frame_similarities):
                frame_gradient = np.abs(np.gradient(
                    savgol_filter(frame_similarities[start-1:end+1], 
                                min(15, (end-start+2)//2*2+1), 3)
                ))[1:-1]  # Remove edge effects
            else:
                # Handle edge cases
                frame_section = frame_similarities[max(0, start-1):min(len(frame_similarities), end+1)]
                smooth_window = min(15, len(frame_section)//2*2+1)
                if smooth_window < 3:
                    smooth_window = 3
                frame_gradient = np.abs(np.gradient(
                    savgol_filter(frame_section, smooth_window, 3)
                ))
                if start > 0:
                    frame_gradient = frame_gradient[1:]
                if end < len(frame_similarities):
                    frame_gradient = frame_gradient[:-1]
            
            # Normalize scores
            if len(smoothed_text) > 0 and len(frame_gradient) > 0:
                text_norm = (smoothed_text - np.min(smoothed_text)) / max(1e-10, np.max(smoothed_text) - np.min(smoothed_text))
                
                if np.max(frame_gradient) - np.min(frame_gradient) > 1e-10:
                    gradient_norm = (frame_gradient - np.min(frame_gradient)) / (np.max(frame_gradient) - np.min(frame_gradient))
                else:
                    gradient_norm = np.zeros_like(frame_gradient)
                
                # Combine scores with weighting
                frame_scores = text_weight * text_norm + (1 - text_weight) * gradient_norm
            else:
                # Fallback if arrays are empty
                frame_scores = np.linspace(0, 1, end - start)
            
            # Get distributed frames by dividing clip into equal segments
            for i in range(frames_to_extract):
                segment_start = start + int(i * (end - start) / frames_to_extract)
                segment_end = start + int((i + 1) * (end - start) / frames_to_extract)
                
                # Ensure we have at least one frame in segment
                if segment_end <= segment_start:
                    segment_end = segment_start + 1
                
                # Find best frame in this segment
                segment_scores = frame_scores[segment_start-start:segment_end-start]
                if len(segment_scores) > 0:
                    best_idx = segment_start + np.argmax(segment_scores)
                    keyframes.append(int(best_idx))
    
    # Remove any duplicate frames (just in case)
    keyframes = sorted(list(set(keyframes)))
    
    # If we still don't have enough frames, add some from highest composite scores
    if len(keyframes) < num_frames:
        # Find frames with high scores that aren't already selected
        remaining = num_frames - len(keyframes)
        candidate_indices = np.argsort(composite_score)[::-1]
        additional = [i for i in candidate_indices if i not in keyframes][:remaining]
        keyframes.extend(additional)
    
    # If we have too many frames, keep the ones with highest scores
    elif len(keyframes) > num_frames:
        # Score each keyframe
        keyframe_scores = [composite_score[i] for i in keyframes]
        # Keep top N frames by score
        top_indices = np.argsort(keyframe_scores)[::-1][:num_frames]
        keyframes = [keyframes[i] for i in top_indices]
    
    return subclips, sorted(keyframes)

def get_n_most_interesting_frames(doc_id, n_frames, video_length, task_type, method="combined", adaptive_sampling_method_max_frames=32, use_subclip_detection=False, use_aks=False):
    """
    Get the n most interesting frames from a video.
    """

    if "mvbench" in task_type:
        clip_embeddings_path = task_to_clip_cache_path["mvbench"] + f"{task_type}"
    elif "tempcompass" in task_type:
        clip_embeddings_path = task_to_clip_cache_path["tempcompass"] + f"{task_type}"
    else:
        clip_embeddings_path = task_to_clip_cache_path[task_type]

    if video_length not in num_frames_to_skip_value:
        suffix = num_frames_to_skip_value[1600]
    else:
        suffix = num_frames_to_skip_value[video_length]

    ############ Temp - remove when all tasks are updated ################
    # clip_embeddings_path = clip_embeddings_path.replace("clip_cache_similarity", "clip_cache_similarity_regclip")
    #######################################################################


    # load an npz file to np array
    text_similarities = np.load(f"{clip_embeddings_path}/cross_sim_{suffix}/{doc_id}.npz")["data"].flatten()

    frame_similarities = np.load(f"{clip_embeddings_path}/image_sim_{suffix}/{doc_id}.npz")["data"].flatten()
    # frame_similarities = np.insert(frame_similarities, 0, 1)

    #convert to float32
    text_similarities = text_similarities.astype(np.float32)
    frame_similarities = frame_similarities.astype(np.float32)

    # get the top n frames
    # key_frames_regions = extract_key_frames(frame_similarities, text_similarities, n_frames=n_frames)["keyframes"]

    if use_aks:
        key_frames_regions = aks_select_frame_indices([text_similarities], adaptive_sampling_method_max_frames, ratio=1, t1=0.4, t2=-100, all_depth=5)[0]
    else:
        if not use_subclip_detection:
            key_frames_regions = extract_key_frames(frame_similarities, text_similarities, n=adaptive_sampling_method_max_frames, method=method)
        else:
            _, key_frames_regions = detect_and_extract_distributed_keyframes(frame_similarities, text_similarities, adaptive_sampling_method_max_frames, 15, 0.8, 1)
    
    scaling_factor = video_length / len(text_similarities)
    
    out = []
    for frame in key_frames_regions:
        out.append(int(frame * scaling_factor))

    return out

# def get_n_most_interesting_frames(doc_id, n_frames, video_length, task_type, method="combined", adaptive_sampling_method_max_frames=32, use_subclip_detection=False):
#     """
#     Get the n most interesting frames from a video.
#     """
#     # try:
#     if task_type != "videomme":
#         task_type = "mlvu"
#     # Load the video
#     clip_embeddings_path = f"/nethome/bdevnani3/flash/lmms_eval_cache/clip_similarity/{task_type}/1600"

#     # load an npz file to np array
#     text_similarities = np.load(f"{clip_embeddings_path}/cross_sim/{doc_id}.npz")["data"].flatten()

#     frame_similarities = np.load(f"{clip_embeddings_path}/image_sim/{doc_id}.npz")["data"].flatten()
#     frame_similarities = np.insert(frame_similarities, 0, 1)

#     #convert to float32
#     text_similarities = text_similarities.astype(np.float32)
#     frame_similarities = frame_similarities.astype(np.float32)

#     # get the top n frames
#     # key_frames_regions = extract_key_frames(frame_similarities, text_similarities, n_frames=n_frames)["keyframes"]

#     if not use_subclip_detection:
#         key_frames_regions = extract_key_frames(frame_similarities, text_similarities, n=adaptive_sampling_method_max_frames, method=method)
#     else:
#         _, key_frames_regions = detect_and_extract_distributed_keyframes(frame_similarities, text_similarities, adaptive_sampling_method_max_frames, 15, 0.8, 1)
    
#     scaling_factor = video_length / len(text_similarities)
    
#     out = []
#     for frame in key_frames_regions:
#         out.append(int(frame * scaling_factor))

#     return out
    # except Exception as e:
    #     print(f"Error in get_n_most_interesting_frames: {e}")
    #     return []
    
def pick_non_overlapping_integers(existing_nums, n, k):
    available = list(set(range(n+1)) - set(existing_nums))
    return random.sample(available, k)

import heapq
import numpy as np

############################## AKS ##########################################

def meanstd(len_scores, dic_scores, n, fns, t1, t2, all_depth):
    split_scores = []
    split_fn = []
    no_split_scores = []
    no_split_fn = []

    for dic_score, fn in zip(dic_scores, fns):
        score = dic_score['score']
        depth = dic_score['depth']
        mean = np.mean(score)
        std = np.std(score)
        top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
        top_score = [score[t] for t in top_n]
        mean_diff = np.mean(top_score) - mean

        if mean_diff > t1 and std > t2:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
        elif depth < all_depth:
            mid = len(score) // 2
            score1 = score[:mid]
            score2 = score[mid:]
            fn1 = fn[:mid]
            fn2 = fn[mid:]
            split_scores.append(dict(score=score1, depth=depth+1))
            split_scores.append(dict(score=score2, depth=depth+1))
            split_fn.append(fn1)
            split_fn.append(fn2)
        else:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)

    if len(split_scores) > 0:
        all_split_score, all_split_fn = meanstd(len_scores, split_scores, n, split_fn, t1, t2, all_depth)
    else:
        all_split_score = []
        all_split_fn = []

    all_split_score = no_split_scores + all_split_score
    all_split_fn = no_split_fn + all_split_fn

    return all_split_score, all_split_fn

def aks_select_frame_indices(all_scores, max_num_frames=64, ratio=1, t1=0.8, t2=-100, all_depth=5):
    """
    all_scores: List[List[float]] – similarity scores per frame for each video
    Returns: List[List[int]] – indices of selected frames for each video
    """
    selected_indices_per_video = []

    for scores in all_scores:
        # Downsample
        sampled_scores = scores[::ratio]
        sampled_indices = list(range(len(scores)))[::ratio]

        if len(sampled_scores) >= max_num_frames:
            norm_scores = (np.array(sampled_scores) - np.min(sampled_scores)) / (np.max(sampled_scores) - np.min(sampled_scores) + 1e-8)
            score_dict = dict(score=norm_scores.tolist(), depth=0)
            all_segs, all_fns = meanstd(len(sampled_scores), [score_dict], max_num_frames, [sampled_indices], t1, t2, all_depth)

            selected = []
            for seg, idx_list in zip(all_segs, all_fns):
                frames_to_select = int(max_num_frames / (2 ** seg['depth']))
                topk = heapq.nlargest(frames_to_select, range(len(seg['score'])), seg['score'].__getitem__)
                selected.extend([idx_list[i] for i in topk])

            selected = sorted(set(selected))
        else:
            selected = sampled_indices

        selected_indices_per_video.append(selected)

    return selected_indices_per_video

############################## AKS ##########################################