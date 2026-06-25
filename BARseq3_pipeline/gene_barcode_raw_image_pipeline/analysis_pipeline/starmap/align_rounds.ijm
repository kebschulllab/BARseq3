#@ int round_num
#@ int round_align
#@ String filepath

setBatchMode('hide');
for (r=1;r<round_num+1;r++) {
	if (r==round_align)
			continue;
	filepath_round = filepath+"round"+r+"/";
	list = getFileList(filepath_round);
	for (i=1; i<list.length; i++) {
			if (indexOf(list[i], "_chcorr") > 0) {
						print(list[i]);
						align_baser = replace(list[i], "r"+r,"r"+round_align);
						align_image_1 = filepath+"round"+round_align+"/"+align_baser;
						align_image_2 = filepath_round + list[i];

						temp = replace(align_baser, "data_chalign_r"+round_align+"_", "");
						fov = replace(temp, "_chcorr.tiff", "");
						direct_transf = filepath_round + "transf/roalign_direct_transf_" + fov + ".txt";
						inverse_transf = filepath_round + "transf/roaligninverse_transf_" + fov + ".txt";

						print(align_image_1);
						print(align_image_2);
						open(align_image_1);
						print("TS");
						rename("1.tiff");
						run("Z Project...", "projection=[Max Intensity]");
						rename("1_merge.tiff");
						getStatistics(area, mean, min, max, std, histogram);
						print(min);
						print(2000);
						setMinAndMax(min, 2550);
						open(align_image_2);
						rename("2.tiff");
						run("Z Project...", "projection=[Max Intensity]");
						rename("2_merge.tiff");
						getStatistics(area, mean, min, max, std, histogram);
						print(min);
						print(2000);
						setMinAndMax(min,2550);
						run("Extract SIFT Correspondences", "source_image=2_merge.tiff target_image=1_merge.tiff initial_gaussian_blur=1.6 steps_per_scale_octave=3 minimum_image_size=288 maximum_image_size=1152 feature_descriptor_size=4 feature_descriptor_orientation_bins=4 closest/next_closest_ratio=0.92 filter maximal_alignment_error=250 minimal_inlier_ratio=0.02 minimal_number_of_inliers=4 expected_transformation=Rigid");
						run("bUnwarpJ", "source_image=2_merge.tiff target_image=1_merge.tiff registration=Accurate image_subsample_factor=0 initial_deformation=[Very Coarse] final_deformation=[Fine] divergence_weight=0 curl_weight=0 landmark_weight=0.9 image_weight=0.9 consistency_weight=10 stop_threshold=0.01 save_transformations save_direct_transformation=["+direct_transf+"] save_inverse_transformation=["+inverse_transf+"]");
						close("*");
						open(align_image_2);
						rename("align.tiff");
						run("Split Channels");
						call("bunwarpj.bUnwarpJ_.loadElasticTransform", direct_transf, "C1-align.tiff", "C1-align.tiff");
						call("bunwarpj.bUnwarpJ_.loadElasticTransform", direct_transf, "C2-align.tiff", "C2-align.tiff");
						call("bunwarpj.bUnwarpJ_.loadElasticTransform", direct_transf, "C3-align.tiff", "C3-align.tiff");
						call("bunwarpj.bUnwarpJ_.loadElasticTransform", direct_transf, "C4-align.tiff", "C4-align.tiff");
						run("Merge Channels...", "c1=C1-align.tiff c2=C2-align.tiff c3=C3-align.tiff c4=C4-align.tiff create");
						saveAs("Tiff", filepath_round + replace(list[i], "_chcorr", "_roundcorr"));
						close("*");
			}
				
	}
}

	
