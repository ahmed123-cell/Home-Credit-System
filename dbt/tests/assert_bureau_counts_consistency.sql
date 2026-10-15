-- Active loans and closed loans cannot exceed total bureau loans reported
select
    sk_id_curr,
    bureau_cnt,
    bureau_active_cnt,
    bureau_closed_cnt
from {{ ref('int_bureau_agg') }}
where (bureau_active_cnt + bureau_closed_cnt) > bureau_cnt
