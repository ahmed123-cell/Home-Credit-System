select
    sk_id_curr,
    count(*)                                                  as pos_rows,
    count(distinct sk_id_prev)                                as pos_prev_nunique,
    min(months_balance)                                       as pos_months_min,
    avg(cnt_instalment)                                       as pos_cnt_instalment_mean,
    avg(cnt_instalment_future)                                as pos_cnt_instalment_future_mean,
    count(*) filter (where name_contract_status = 'Active')    as pos_active_rows,
    count(*) filter (where name_contract_status = 'Completed') as pos_completed_rows,
    max(sk_dpd)                                               as pos_dpd_max,
    avg(sk_dpd)                                               as pos_dpd_mean,
    avg((sk_dpd > 0)::int)                                    as pos_dpd_share,
    max(sk_dpd_def)                                           as pos_dpd_def_max,
    avg((sk_dpd_def > 0)::int)                                as pos_dpd_def_share,
    avg(sk_dpd) filter (where months_balance >= -12)          as pos_dpd_mean_1y
from {{ ref('stg_pos_cash_balance') }}
group by sk_id_curr