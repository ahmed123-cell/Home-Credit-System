with base as (
    select
        *,
        amt_balance::double precision / nullif(amt_credit_limit_actual, 0) as utilization
    from {{ ref('stg_credit_card_balance') }}
)

select
    sk_id_curr,
    count(*)                                   as cc_rows,
    count(distinct sk_id_prev)                 as cc_prev_nunique,
    min(months_balance)                        as cc_months_min,

    avg(amt_balance)                           as cc_balance_mean,
    max(amt_balance)                           as cc_balance_max,
    avg(amt_credit_limit_actual)               as cc_limit_mean,
    avg(utilization)                           as cc_utilization_mean,
    max(utilization)                           as cc_utilization_max,
    avg(utilization) filter (where months_balance >= -6) as cc_utilization_mean_6m,

    avg(amt_drawings_current)                  as cc_drawings_mean,
    avg(amt_drawings_atm_current)              as cc_drawings_atm_mean,
    avg(cnt_drawings_current)                  as cc_cnt_drawings_mean,
    avg(amt_payment_current)                   as cc_payment_mean,
    avg(amt_payment_total_current)             as cc_payment_total_mean,
    avg(amt_inst_min_regularity)               as cc_min_regularity_mean,
    avg(amt_total_receivable)                  as cc_receivable_mean,

    max(sk_dpd)                                as cc_dpd_max,
    avg((sk_dpd > 0)::int)                     as cc_dpd_share,
    max(sk_dpd_def)                            as cc_dpd_def_max,
    avg((sk_dpd_def > 0)::int)                 as cc_dpd_def_share
from base
group by sk_id_curr