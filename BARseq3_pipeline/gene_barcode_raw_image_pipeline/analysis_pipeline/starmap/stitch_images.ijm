#@ String filepath

run("Grid/Collection stitching", "type=[Positions from file] order=[Defined by TileConfiguration] directory=["+filepath+"] layout_file=positions.txt fusion_method=Average regression_threshold=0.10 max/avg_displacement_threshold=250 absolute_displacement_threshold=250 compute_overlap ignore_z_stage subpixel_accuracy display_fusion computation_parameters=[Save memory (but be slower)] image_output=[Write to disk] output_directory="+filepath);
