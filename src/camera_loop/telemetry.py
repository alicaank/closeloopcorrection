"""Separate observer acceptance, controller calls and actual command changes."""
import numpy as np


def summarize_controller(controller, legacy_events=()):
    events=controller.get('controller_events',[])
    accepted=[e for e in events if e.get('accepted') and e.get('confidence',0)>0]
    return dict(boundaries=len(events),accepted=len(accepted),
        accepted_feedback_boundaries=len(accepted),
        applied_feedback=len(accepted),applied_feedback_semantics='accepted positive-confidence updates; not scheduled calls',
        scheduled_calls=len(events),
        legacy_reported_applied=sum(bool(e.get('correction_applied')) for e in legacy_events),
        fresh_nonzero_feedback=sum(bool(e.get('fresh_residual_feedback')) for e in events) if all('fresh_residual_feedback' in e for e in events) else None,
        changed_command_boundaries=sum(bool(e.get('commands_changed')) for e in events) if all('commands_changed' in e for e in events) else None,
        fit_boundaries=sum(any(e['response_update_axes']) for e in events),
        active_gain_boundaries=sum(not np.allclose(e['response_gains'],1.,atol=1e-8,rtol=0) for e in events),
        clipped_boundaries=sum(bool(np.any(e['clipped'])) for e in events),
        final_gains=events[-1]['response_gains'] if events else [1.]*6)
