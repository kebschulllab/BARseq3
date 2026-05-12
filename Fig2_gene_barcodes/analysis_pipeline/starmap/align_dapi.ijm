#@ int fov_num
#@ int dapi_round
#@ String filepath
#@ String subfolder
print("align dapi/nissl iamges with the transformation from round alignment")
//setBatchMode('hide');

for (r=0;r<fov_num;r++) {
	filepath_round = filepath + subfolder;

	list = getFileList(filepath_round);
	print(filepath_round);
	for (i=0; i<list.length; i++) {
			if (indexOf(list[i], "_raw") > 0) {
						print(list[i]);
						align_baser = replace(list[i], "fov" + r, "fov0");
						print(align_baser);
						align_image_1 = filepath_round + align_baser;
						print(align_image_1);
						align_image_2 = filepath_round + list[i];
						print(align_image_2);
						
						
						temp = replace(align_baser, "align_", "");
						print(temp);
						fov = replace(temp, "_raw.tiff", "");
						print(fov);

						open(align_image_1);
						rename("align_image_1.tiff");
						print("TS");
						
						open(align_image_2);
						rename("align_image_2.tiff");
						print("TS");
						
						filepath_transformation = filepath + "round" + dapi_round + "/transf";
						direct_transf = filepath_transformation + "/roalign_direct_transf_" + fov + ".txt";
						print(direct_transf);		

						call("bunwarpj.bUnwarpJ_.loadElasticTransform", direct_transf, "align_image_1.tiff", "align_image_2.tiff");
						close("align_image_1.tiff");
						saveAs("Tiff", filepath_round + replace(list[i], "_raw", "_aligned"));
						close("*");
			}
				
	}
}