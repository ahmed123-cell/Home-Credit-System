with base as (
    select
        sk_id_curr,
        sk_id_prev,
        days_instalment,
        amt_instalment,
        amt_payment,
        greatest(days_entry_payment - days_instalment, 0) as dpd,
        greatest(days_instalment - days_entry_payment, 0) as dbd,
        amt_payment::double precision / nullif(amt_instalment, 0) as payment_ratio,
        amt_instalment - amt_payment                      as payment_diff
    from {{ ref('stg_installments_payments') }}
)

select
    sk_id_curr,
    count(*)                                             as inst_cnt,
    count(distinct sk_id_prev)                           as inst_prev_nunique,

    avg(dpd)                                             as inst_dpd_mean,
    max(dpd)                                             as inst_dpd_max,
    sum(dpd)                                             as inst_dpd_sum,
    avg((dpd > 0)::int)                                  as inst_late_share,
    avg(dbd)                                             as inst_dbd_mean,
    max(dbd)                                             as inst_dbd_max,

    avg(payment_ratio)                                   as inst_payment_ratio_mean,
    min(payment_ratio)                                   as inst_payment_ratio_min,
    sum(payment_diff)                                    as inst_payment_diff_sum,
    avg((payment_diff > 0)::int)                         as inst_underpaid_share,
    sum(amt_instalment)                                  as inst_amt_instalment_sum,
    sum(amt_payment)                                     as inst_amt_payment_sum,

    -- recent behavior (last 12 months)
    avg(dpd) filter (where days_instalment >= -365)      as inst_dpd_mean_1y,
    avg((dpd > 0)::int) filter (where days_instalment >= -365) as inst_late_share_1y,
    avg(payment_ratio) filter (where days_instalment >= -365)  as inst_payment_ratio_mean_1y
from base
group by sk_id_curr