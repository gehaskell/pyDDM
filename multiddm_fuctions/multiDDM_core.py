
# Starting with actual DDM calculation line 110

max_tau = 100

results = [{"box_num":1}, {"box_num":2}]
ind_frames = []


for tau in range(1, max_tau+1):

    for box_result in results:
        results['accum_abs_FT_diff_image'] =zeros([boxsize_vector(rc),boxsize_vector(rc),N_boxes(rc)], 'single')

    ind_frames = choose_ind_frames(tau, N_couple_frames_to_average, N_frames);
    ccc = numel(ind_frames); # number of couples of frames at this tau

    for i in ind_frames:
        # take difference
        tempdiff = single(frame_stack(:,:,i)) - single(frame_stack(:,:,i+tau))

        # now take the right fft of the right bit for each box
        for rc in range(len(results)):
            tempdiff_chunk = tempdiff(row_offset(rc):row_offset(rc) + row_span(rc) - 1,
                                      col_offset(rc): col_offset(rc) + col_span(rc) - 1)

            # reshape into 3d stack for fast fft2
            tempdiff_stack = reshape(permute(reshape(tempdiff_chunk, row_span(rc), boxsize_vector(rc), []), [2 1 3]),
                                     boxsize_vector(rc), boxsize_vector(rc), []);

            # fft2 of difference image
            ft_tempdiff = fft2(tempdiff_stack) * fft2_norm_factor(rc);

            DRes(rc).accum_abs_FT_diff_image = ...
            DRes(rc).accum_abs_FT_diff_image + ...
            real(ft_tempdiff). ^ 2 + imag(ft_tempdiff). ^ 2;

    for rc = 1:N_boxsizes
        DRes(rc).averaged_abs_FT_diff_image = ...
        DRes(rc).accum_abs_FT_diff_image. / ccc;

        for bc = 1:N_boxes(rc)


            oneD_power_spectrum = accumarray(DRes(rc).distance_map, reshape(DRes(rc).averaged_abs_FT_diff_image(:,:, bc), [], 1) )./ ...
                            DRes(rc).dist_counts; # radial average

            # fill each column of the output with the 1D power spectrum just calculated
            Res(rc).Box(bc).Iqtau(:, tau) = oneD_power_spectrum(2: max_q(rc) + 1);


    cccount(tau) = ccc;
