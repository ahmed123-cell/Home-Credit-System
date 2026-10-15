-- Approved, refused, and canceled contract counts cannot exceed total previous applications
select
    sk_id_curr,
    prev_cnt,
    prev_approved_cnt,
    prev_refused_cnt,
    prev_canceled_cnt
from {{ ref('int_previous_application_agg') }}
where (prev_approved_cnt + prev_refused_cnt + prev_canceled_cnt) > prev_cnt
