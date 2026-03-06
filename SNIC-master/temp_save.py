'''
def main(image_path):
    downsized_image, original_image = downsize_image(image_path)
    
    downsized_array = np.array(downsized_image)

    # Step 1: Apply SLIC segmentation
    slic_downsize = apply_slic_segmentation(downsized_array)
    
    slic_original = resize_segmentation_to_original(slic_downsize, original_image.size)

    #slic_downsize, labels = segment(image_path, 250, 5, True)

    num_superpixels1 = len(np.unique(slic_downsize))
    #print("Number of superpixels in slic_downsize:", num_superpixels1)
    
    #slic_original = resize_segmentation_to_original(slic_downsize, original_image.size)

    result1 = calculate_superpixel_mean(slic_downsize, downsized_array)
    
    #print("The mean of each superpixel is:")
    #print(result1)

    

    # Step 3: Extract segments and batch features
    segments, segment_dict = extract_segments(original_image, slic_original)
    if not segments:
        print("No valid segments found.")
        return

    features_batch = extract_features_batch(segments)
    features_dict = {segment_id: features_batch[i] for i, segment_id in enumerate(segment_dict)}

    #print(len(features_batch[0]))



    # Step 3: Create a copy of the SLIC segmentation for modification
    slic_downsize_copy = np.copy(slic_downsize)

    # Step 4: Merge segments based on cosine similarity

    

    
    iteration = 5

    for i in range(iteration):
        unique_segments = np.unique(slic_downsize_copy)  # Refresh unique segments at the start of each iteration
    
        for segment_id in unique_segments:
            if segment_id not in features_dict:
               continue
        
            segment_feature = features_dict[segment_id]
            neighbors = find_neighbors(slic_downsize_copy, segment_id)

            for neighbor_id in neighbors:
                if neighbor_id not in features_dict:
                   continue

                neighbor_feature = features_dict[neighbor_id]
                similarity = cosine_similarity([segment_feature], [neighbor_feature])[0, 0]

                if similarity >= similarity_threshold:
                # Merge neighbor_id into segment_id
                   slic_downsize_copy[slic_downsize_copy == neighbor_id] = segment_id
                   print(f"Merged {neighbor_id} into {segment_id}")

    # Step 5: Apply half-transparent color mask to the downsized image
        num_superpixels = len(np.unique(slic_downsize_copy))
        print("Number of superpixels in slic_downsize_copy:", num_superpixels)

# Update the original segmentation after all iterations
    slic_downsize = np.copy(slic_downsize_copy)

    prediction_path = '/home/swz45/Desktop/sample_test_img/Big_RockfordBeach_032924_EGN_shadow_3cm_rect_wcr_ss_star_00027_horizon_raw.png'

    prediction_pil = Image.open(prediction_path).convert("L")

    prediction = np.array(prediction_pil)

    modified_prediction = majority_vote(prediction, slic_downsize_copy)
    
    output_path = "/home/swz45/Desktop/sample_test_img/Big_RockfordBeach_032924_EGN_shadow_3cm_rect_wcr_ss_star_00027_slic.png"
    cv2.imwrite(output_path, modified_prediction)

    print(f"Updated grayscale prediction saved at {output_path}")

    colorized_prediction = grayscale_to_color(modified_prediction, class_colors)

    # Save the colorized image
    output_color_path = "/home/swz45/Desktop/sample_test_img/Big_RockfordBeach_032924_EGN_shadow_3cm_rect_wcr_ss_star_00027_sliccolor_cp5_sp250.png"
    cv2.imwrite(output_color_path, cv2.cvtColor(colorized_prediction, cv2.COLOR_RGB2BGR))

    #Computing the overall accuracy
    ground_truth_file = '/home/swz45/Desktop/sample_test_img/Big_RockfordBeach_032924_EGN_shadow_3cm_rect_wcr_ss_star_00027_horizon_gt_msk.png'
    
    ground_truth = Image.open(ground_truth_file).convert("L")

    gt = np.array(ground_truth)

    acc = compute_accuracy(modified_prediction, gt)

    print("The accuracy after majority voting is:")

    print(acc)
    

    #print(np.where(slic_downsize_copy == 1))
    apply_transparent_overlay(downsized_image, slic_downsize,slic_downsize_copy, output_path='/home/swz45/Desktop/sample_test_img/Big_RockfordBeach_032924_bgremove_cp5_sp250.jpg')
'''

    
'''
if __name__ == "__main__":

   
    


    image_path = "/home/swz45/Desktop/sample_test_img/Big_RockfordBeach_032924_EGN_shadow_3cm_rect_wcr_ss_star_00027_horizonrgb.png"
    main(image_path)
'''