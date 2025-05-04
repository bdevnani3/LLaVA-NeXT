from transformers import AutoConfig

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d
import heapq
import random





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


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d
import heapq

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


def get_n_most_interesting_frames(doc_id, n_frames, video_length, task_type):
    """
    Get the n most interesting frames from a video.
    """
    try:
        if task_type != "videomme":
            task_type = "mlvu"
        # Load the video
        clip_embeddings_path = f"/nethome/bdevnani3/flash/lmms_eval_cache/clip_similarity/{task_type}/1600"

        # load an npz file to np array
        text_similarities = np.load(f"{clip_embeddings_path}/cross_sim/{doc_id}.npz")["data"].flatten()

        frame_similarities = np.load(f"{clip_embeddings_path}/image_sim/{doc_id}.npz")["data"].flatten()
        frame_similarities = np.insert(frame_similarities, 0, 1)

        #convert to float32
        text_similarities = text_similarities.astype(np.float32)
        frame_similarities = frame_similarities.astype(np.float32)
        

        # get the top n frames
        key_frames_regions = extract_key_frames(frame_similarities, text_similarities, 
                                            n=n_frames, method='regions')
        
        scaling_factor = video_length / len(text_similarities)
        
        out = []
        for frame in key_frames_regions:
            out.append(int(frame * scaling_factor))


        return out
    except Exception as e:
        print(f"Error in get_n_most_interesting_frames: {e}")
        return []
    
def pick_non_overlapping_integers(existing_nums, n, k):
    available = list(set(range(n+1)) - set(existing_nums))
    return random.sample(available, k)