#@ int round_num
#@ int fov_align
#@ String filepath

setBatchMode('hide');
for (r=1;r<round_num+1;r++) {
	filepath_round = filepath+"round"+r+"/";
	direct_transf = filepath_round + "transf/chalign_direct_transf.txt";
	inverse_transf = filepath_round + "transf/chalign_inverse_transf.txt";

	align_image_c1 = filepath_round+"data_chalign_r"+r+"_fov"+fov_align+"_c0.tiff"; 
	open(align_image_c1);
	rename("1.tiff");
	setMinAndMax(0, 2500);
	
	align_image_c2 = filepath_round+"data_chalign_r"+r+"_fov"+fov_align+"_c1.tiff";
	open(align_image_c2);
	rename("2.tiff");
	setMinAndMax(0, 2500);
	
	run("Extract SIFT Correspondences", "source_image=2.tiff target_image=1.tiff initial_gaussian_blur=1.6 steps_per_scale_octave=3 minimum_image_size=288 maximum_image_size=1152 feature_descriptor_size=4 feature_descriptor_orientation_bins=4 closest/next_closest_ratio=0.92 filter maximal_alignment_error=25 minimal_inlier_ratio=0.05 minimal_number_of_inliers=7 expected_transformation=Perspective");
	run("bUnwarpJ", "source_image=2.tiff target_image=1.tiff registration=Accurate image_subsample_factor=0 initial_deformation=[Very Coarse] final_deformation=[Very Fine] divergence_weight=0 curl_weight=0 landmark_weight=0.9 image_weight=1 consistency_weight=10 stop_threshold=0.02 save_transformations save_direct_transformation=["+direct_transf+"] save_inverse_transformation=["+inverse_transf+"]");
	
	list = getFileList(filepath_round);
	close("*");
	for (i=0; i<list.length; i++) {
		if (indexOf(list[i], "_c1") > 0) {
			print(list[i]);
			temp_name = replace(list[i], "_c1.tiff", "");
			open(filepath_round+temp_name+"_c0.tiff");
			open(filepath_round+temp_name+"_c1.tiff");
			open(filepath_round+temp_name+"_c2.tiff");
			open(filepath_round+temp_name+"_c3.tiff");
			call("bunwarpj.bUnwarpJ_.loadElasticTransform", direct_transf, temp_name+"_c0.tiff", temp_name+"_c1.tiff");
			call("bunwarpj.bUnwarpJ_.loadElasticTransform", direct_transf, temp_name+"_c2.tiff", temp_name+"_c3.tiff");
			selectWindow(temp_name+"_c0.tiff");
			run("32-bit");
			selectWindow(temp_name+"_c2.tiff");
			run("32-bit");
			run("Merge Channels...", "c1="+temp_name+"_c0.tiff c2="+temp_name+"_c1.tiff c3="+temp_name+"_c2.tiff c4="+temp_name+"_c3.tiff create");
			saveAs("Tiff", filepath_round + temp_name +"_chcorr.tiff");
			close("*");
		}
	}
}	
