def _generate_causal_fast(self,
                          input_prompt,
                          img,
                          action_path,
                          chunk_size=3,
                          max_area=480 * 832,
                          frame_num=81,
                          timesteps_index=[0, 179, 358, 679],
                          shift=5.0,
                          seed=-1,
                          offload_model=True,
                          max_sequence_length=512,
                          max_attention_size=None,):
    r"""
    Generates video frames from input image and text prompt using diffusion process.

    Args:
        input_prompt (`str`):
            Text prompt for content generation.
        img (PIL.Image.Image):
            Input image tensor. Shape: [3, H, W]
        max_area (`int`, *optional*, defaults to 720*1280):
            Maximum pixel area for latent space calculation. Controls video resolution scaling
        frame_num (`int`, *optional*, defaults to 81):
            How many frames to sample from a video. The number should be 4n+1
        shift (`float`, *optional*, defaults to 5.0):
            Noise schedule shift parameter. Affects temporal dynamics
            [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
        sample_solver (`str`, *optional*, defaults to 'unipc'):
            Solver used to sample the video.
        sampling_steps (`int`, *optional*, defaults to 40):
            Number of diffusion sampling steps. Higher values improve quality but slow generation
        seed (`int`, *optional*, defaults to -1):
            Random seed for noise generation. If -1, use random seed
        offload_model (`bool`, *optional*, defaults to True):
            If True, offloads models to CPU during generation to save VRAM

    Returns:
        torch.Tensor:
            Generated video frames tensor. Dimensions: (C, N H, W) where:
            - C: Color channels (3 for RGB)
            - N: Number of frames (81)
            - H: Frame height (from max_area)
            - W: Frame width from max_area)
    """

    if input_prompt is not None and isinstance(input_prompt, str):
        batch_size = 1
    elif input_prompt is not None and isinstance(input_prompt, list):
        batch_size = len(input_prompt)
    else:
        batch_size = 1

    assert action_path is not None, "action_path is required"
    c2ws = np.load(os.path.join(action_path, "poses.npy")) # opencv coordinate
    len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
    frame_num = ((frame_num - 1) // 4) * 4 + 1
    frame_num = min(frame_num, len_c2ws)
    c2ws = c2ws[:frame_num]

    # preprocess
    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

    F = frame_num
    h, w = img.shape[1:]
    aspect_ratio = h / w
    lat_h = round(
        np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
        self.patch_size[1] * self.patch_size[1])
    lat_w = round(
        np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
        self.patch_size[2] * self.patch_size[2])
    h = lat_h * self.vae_stride[1]
    w = lat_w * self.vae_stride[2]
    lat_f = (F - 1) // self.vae_stride[0] + 1
    lat_f = int(lat_f - (lat_f % chunk_size))
    F = (lat_f - 1) * 4 + 1
    max_seq_len = chunk_size * lat_h * lat_w // (
        self.patch_size[1] * self.patch_size[2])
    max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
    # Reset per-generate state: cross-attn K/V cache will be freshly
    # initialized below; the first DiT forward must compute and store.
    self._cross_attn_initialized = False

    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)
    noise = torch.randn(
        16,
        lat_f,
        lat_h,
        lat_w,
        dtype=torch.float32,
        generator=seed_g,
        device=self.device)

    msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
    msk[:, 1:] = 0
    msk = torch.concat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
    ],
                       dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]

    # 2. Prepare timesteps
    self.scheduler.set_timesteps(self.num_train_timesteps, shift=shift)
    timesteps = self.scheduler.timesteps[timesteps_index]

    # preprocess
    # T5 cache: skip the encoder entirely if we've seen this exact prompt
    # before in this pipe instance. Bit-identical: cached tensor is the
    # same object returned by the prior call.
    cache_key = hashlib.sha256(input_prompt.encode('utf-8')).hexdigest()
    if cache_key in self._t5_cache:
        context = self._t5_cache[cache_key]
    else:
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
        self._t5_cache[cache_key] = context

    Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()

    # The provided intrinsics are for original image size (480p). We need to transform them according to the new image size (h, w).
    Ks = get_Ks_transformed(Ks,
                            height_org=480,
                            width_org=832,
                            height_resize=h,
                            width_resize=w,
                            height_final=h,
                            width_final=w)
    Ks = Ks[0]

    len_c2ws = len(c2ws)
    len_c2ws_ = int((len_c2ws - 1) // 4) + 1
    len_c2ws_ = int(len_c2ws_ - (len_c2ws_ % chunk_size))
    c2ws_infer = interpolate_camera_poses(
        src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
        src_rot_mat=c2ws[:, :3, :3],
        src_trans_vec=c2ws[:, :3, 3],
        tgt_indices=np.linspace(0, len_c2ws - 1, len_c2ws_),
    )
    c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
    Ks = Ks.repeat(len(c2ws_infer), 1)

    c2ws_infer = c2ws_infer.to(self.device)
    Ks = Ks.to(self.device)
    wasd_action = None
    c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
    c2ws_plucker_emb = rearrange(
        c2ws_plucker_emb,
        'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
        c1=int(h // lat_h),
        c2=int(w // lat_w),
    )
    c2ws_plucker_emb = c2ws_plucker_emb[None, ...] # [b, f*h*w, c]
    c2ws_plucker_emb = rearrange(c2ws_plucker_emb, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
    if wasd_action is not None:
        wasd_action_tensor = wasd_action[:, None, None, :].repeat(1, h, w, 1) # [f, h, w, 3]
        wasd_action_tensor = rearrange(
            wasd_action_tensor,
            'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
            c1=int(h // lat_h),
            c2=int(w // lat_w),
        )
        wasd_action_tensor = wasd_action_tensor[None, ...] # [b, f*h*w, c]
        wasd_action_tensor = rearrange(wasd_action_tensor, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
        c2ws_plucker_emb = torch.cat([c2ws_plucker_emb, wasd_action_tensor], dim=1)

    y = self.vae.encode([
        torch.concat([
            torch.nn.functional.interpolate(
                img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                    0, 1),
            torch.zeros(3, F - 1, h, w)
        ],
                     dim=1).to(self.device)
    ])[0]
    y = torch.concat([msk, y])

    @contextmanager
    def noop_no_sync():
        yield

    no_sync_model = getattr(self.model, 'no_sync', noop_no_sync)

    # Initialize KV cache to all zeros
    model_args = self.model.config
    transformer_dtype = self.pipe_dtype
    frame_seqlen = int(noise.shape[-2] * noise.shape[-1]// 4)
    if self.local_attn_size > -1:
        kv_size = frame_seqlen * self.local_attn_size
    else:
        kv_size = frame_seqlen * lat_f
    head_dim = model_args.dim // model_args.num_heads
    local_num_heads = model_args.num_heads // self.sp_size
    self_kv_shape = [batch_size, kv_size, local_num_heads, head_dim]
    self_kv_cache = self._initialize_self_kv_cache(num_layers=model_args.num_layers,
                                                   shape=self_kv_shape,
                                                   dtype=transformer_dtype,
                                                   device=self.device)
    cross_kv_shape = [batch_size, max_sequence_length, model_args.num_heads, head_dim]
    cross_kv_cache = self._initialize_crossattn_cache(num_layers=model_args.num_layers,
                                                      shape=cross_kv_shape,
                                                      dtype=transformer_dtype,
                                                      device=self.device)
    # evaluation mode
    with (
            torch.amp.autocast('cuda', dtype=self.param_dtype),
            torch.no_grad(),
            no_sync_model(),
    ):
        # sample videos
        latent = noise
        latents_chunk = latent.split(chunk_size, dim=1) # [c, f, h, w]
        condition_chunk = y.split(chunk_size, dim=1)
        c2ws_plucker_emb_chunk = c2ws_plucker_emb.split(chunk_size, dim=2)
        num_inference_chunk = len(latents_chunk)
        pred_latent_chunks = []
        for chunk_id in tqdm(range(num_inference_chunk)):
            current_latent = latents_chunk[chunk_id]
            current_condition = condition_chunk[chunk_id]
            current_c2ws_plucker_emb = c2ws_plucker_emb_chunk[chunk_id]

            dit_cond_dict = {
                "c2ws_plucker_emb": current_c2ws_plucker_emb.chunk(1, dim=0),
            }

            kwargs = {
                'context': [context[0]],
                'seq_len': max_seq_len,
                'y': [current_condition],
                'dit_cond_dict': dit_cond_dict,
                'kv_cache': self_kv_cache,
                'crossattn_cache': cross_kv_cache,
                'current_start': chunk_id * chunk_size * frame_seqlen,
                'max_attention_size': kv_size if max_attention_size is None else max_attention_size,
                'frame_seqlen': frame_seqlen,
            }

            if offload_model:
                torch.cuda.empty_cache()

            for timestep_idx in range(len(timesteps)):
                latent_model_input = [current_latent.to(self.device)]
                current_timestep = [timesteps[timestep_idx]]

                timestep = torch.stack(current_timestep).to(self.device)

                noise_pred = self.model(
                    x=latent_model_input, t=timestep,
                    cross_attn_first_call=not self._cross_attn_initialized,
                    **kwargs)[0]
                self._cross_attn_initialized = True

                if offload_model:
                    torch.cuda.empty_cache()

                x0 = self._convert_flow_pred_to_x0(
                    flow_pred=noise_pred,
                    xt=current_latent,
                    timestep=current_timestep[0],
                    scheduler=self.scheduler,
                )

                if timestep_idx < len(timesteps) - 1:
                    next_timestep = timesteps[timestep_idx + 1]
                    current_latent = self.scheduler.add_noise(x0, torch.randn(x0.shape, generator=seed_g, device=x0.device, dtype=x0.dtype), next_timestep)
                else:
                    # note return x0
                    break

            pred_latent_chunks.append(x0)

            __import__('memory_hooks').lingbot_cache(locals())
            # Update kv cache
            context_timestep = [timesteps[-1] * 0.0]
            timestep = torch.stack(context_timestep).to(self.device)
            self.model(x=[x0], t=timestep,
                       cross_attn_first_call=False,
                       **kwargs)
            __import__('lingbot_gauge_hook').after_clean(locals())

        pred_latent_chunks = torch.cat(pred_latent_chunks, dim=1)

        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()

        if self.rank == 0:
            videos = self.vae.decode([pred_latent_chunks])

    # del noise, latent, x0
    # del sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return videos[0] if self.rank == 0 else None
