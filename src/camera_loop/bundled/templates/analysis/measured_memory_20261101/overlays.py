"""Auditable, fail-closed source overlays; released drivers remain untouched."""
from pathlib import Path
from runtime import A,R,replace_once

DRIVERS={
    'dreamx':R/'dreamx_run.py',
    'lingbot':A/'new_models_loop_20261028/lingbot_pair.py',
    'sana':A/'new_models_loop_20261028/sana_run.py',
    'worldwarp':A/'new_models_loop_20261028/worldwarp_run.py',
    'matrixgame':A/'continuous_eval_20261029/mg35_stream_run.py',
    'matrix_child':A/'continuous_eval_20261029/mg35_stream_launch.py',
    'lyra':A/'continuous_eval_20261029/lyra_run.py',
    'lyra_child':A/'continuous_eval_20261029/lyra_plugin.py',
    'astronex':A/'long_domain_20261028/astronex_run.py',
}


def native_source(model,source):
    """Used both by runtime installers and source-contract tests."""
    if model=='lingbot':
        needle='            # Update kv cache'
        return replace_once(source,needle,"            __import__('memory_hooks').lingbot_cache(locals())\n"+needle)
    if model=='sana':
        needle='            noise_pred, updated_kv_cache = self.model('
        return replace_once(source,needle,"            __import__('memory_hooks').sana_capture(locals())\n"+needle)
    if model=='astronex':
        needle='            # Step 3.3: rerun with timestep zero to update KV cache using clean context'
        return replace_once(source,needle,"            vm_chunk, prope_window_viewmats = __import__('astronex_hook').memory(locals())\n"+needle)
    if model=='worldwarp':
        needle='        rel_camera_poses[:,:T_video,...] = rel_cache_poses'
        return replace_once(source,needle,"        rel_cache_poses = __import__('memory_hooks').worldwarp_memory(locals())\n"+needle)
    if model=='matrixgame':
        needle='            _observe(section_idx, unwrapped_model.pipe, camera_data)'
        return replace_once(source,needle,needle+"\n            last_predicting_extr = __import__('memory_hooks').matrix_registered(section_idx,last_predicting_extr,clean_history_extr)")
    raise ValueError(model)


def driver_source(model):
    source=DRIVERS[model].read_text()
    source="import sys\nsys.path.insert(0,"+repr(str(R))+")\nimport memory_hooks as mm\nfrom overlays import native_source\n"+source
    if model=='dreamx':
        source=replace_once(source,"    p.add_argument('--delta', type=float, default=0.)", "    p.add_argument('--latents', type=int, default=42)")
        source=replace_once(source,"    p.add_argument('--prefix-latents', type=int, choices=[6,15], required=True)", "    p.add_argument('--prefix-latents', type=int, default=3)")
        source=replace_once(source,"    assert a.delta in [0., -.8, -.4, -.2, .1, .2, .3, .4, .8]", "    a.delta=0.\n    if a.latents<6 or a.latents%3 or a.prefix_latents<3 or a.prefix_latents%3 or a.prefix_latents>=a.latents:raise ValueError('DreamX needs complete 3-latent chunks and a causal first-write boundary')")
        source=replace_once(source,'    total_latents=a.prefix_latents+3','    total_latents=a.latents')
        source=replace_once(source,'    commands = perturb(plan, a.delta, boundary)','    commands = plan.copy()\n    if len(plan)!=frame_count:raise ValueError("Trajectory shorter than rollout")')
        source=replace_once(source,"default='measured')","default=mm.policy())")
    elif model=='lingbot':
        source=replace_once(source,' scope=dict(lib.__dict__);exec(compile(src'," src=native_source('lingbot',src)\n scope=dict(lib.__dict__);exec(compile(src")
        source=replace_once(source,'pipe._loop_observer=obs',"pipe._loop_observer=obs;obs.memory_context=dict(K=K,depth=depth,out=a.out/arm)")
        source=replace_once(source,"'memory_policy':'KV updates retain actual conditioning; no rewriting'","'memory_policy':mm.policy()")
    elif model=='sana':
        source=replace_once(source,'  state.update(pipe=pipe,ray=ray,plucker=plucker,latent_T=latent_T)',
            '  mm.install_sana()\n  state.update(pipe=pipe,ray=ray,plucker=plucker,latent_T=latent_T)')
        memory_call="  if mm.policy()=='measured' and worker is not None:worker.close();worker=None\n  memory_boundary=mm.sana_write(dict(locals(),state=state,commands=commands,prefix=prefix,K=K,depth=depth,lib=lib,intr=intr,a=a))"
        if "  pipe=state['pipe']\n" in source:
            # Memory observation/replay retains its own placement scope. Keep
            # the decoder on GPU until the endpoint scope moves stage-1 first.
            needle="  pipe=state['pipe']\n"
            source=replace_once(source,needle,needle+memory_call+"\n")
        else:
            needle="  pipe=state['pipe'];pipe._move_vae_decoder_for_streaming('cpu');torch.cuda.empty_cache()"
            source=replace_once(source,needle,needle+"\n"+memory_call)
        source=replace_once(source,'(commands[lo]@np.linalg.inv(start))@desired','(memory_boundary@np.linalg.inv(start))@desired')
        source=replace_once(source,"a=ap.parse_args();a.out.mkdir", "a=ap.parse_args();\n if a.arm=='native' and mm.policy()!='commanded':raise ValueError('Use --arm baseline for serialized measured-memory callbacks')\n a.out.mkdir")
    elif model=='worldwarp':
        source=replace_once(source,' scope=dict(geometry.__dict__,_calibrate_native=calibrate_native)',
            " source=native_source('worldwarp',source)\n scope=dict(geometry.__dict__,_calibrate_native=calibrate_native)")
        needle='   path=super().run_inference_chunk('
        source=replace_once(source,needle,"   mm.worldwarp_prepare(dict(locals(),prefix=prefix,K=K,depth=depth,a=a,plan=plan,scale_state=scale_state))\n"+needle)
    elif model in ('matrixgame','lyra'):
        child='matrix_child' if model=='matrixgame' else 'lyra_child'
        old="str(R/'mg35_stream_launch.py')" if model=='matrixgame' else "str(R/'lyra_plugin.py')"
        source=replace_once(source,old,repr(str(R/'launch.py'))+','+repr(child)+",'--'")
    elif model=='matrix_child':
        source=replace_once(source," if section==0 or arm=='native':return"," if section==0 or (arm=='native' and mm.policy()=='commanded'):return")
        source=replace_once(source," commands=plan[lo+1:lo+85].copy()", " memory_boundary=mm.matrix_context(dict(locals(),out=out,all_commands=all_commands,prefix=prefix,K=K,depth=depth))\n commands=plan[lo+1:lo+85].copy()")
        source=replace_once(source,'np.array_equal(all_commands[lo],plan[lo])','np.array_equal(memory_boundary,plan[lo])')
        source=replace_once(source,'else:commands=(all_commands[lo]@np.linalg.inv(start))@desired','else:commands=(memory_boundary@np.linalg.inv(start))@desired')
        source=replace_once(source,'scope=dict(V.__dict__,_observe=observe',"src=native_source('matrixgame',src)\nscope=dict(V.__dict__,_observe=observe")
    elif model=='lyra_child':
        source=replace_once(source,'        _orig_update(self, end_px_idx)','        _orig_update(self, end_px_idx)\n        mm.lyra_memory(self,end_px_idx)')
    elif model=='astronex':
        source=replace_once(source,"    block_done(L, CTX['videos'][CTX['i']], CTX['st'], final)",
            "    if final or CTX['st']['prev'] != int(L['current_start_frame']):\n        block_done(L, CTX['videos'][CTX['i']], CTX['st'], final)")
        source=replace_once(source,"    name = 'inference.pipelines.causal_diffusion_inference';", "    src=native_source('astronex',src)\n    name = 'inference.pipelines.causal_diffusion_inference';")
        source=replace_once(source,"types.SimpleNamespace(hook=hook)","types.SimpleNamespace(hook=hook,memory=lambda L:mm.astronex_memory(L,CTX,hook))")
    compile(source,str(DRIVERS[model]),'exec')
    return source
