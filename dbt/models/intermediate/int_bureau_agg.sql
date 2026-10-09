with joined as (
    select b.*, bb.bb_months_cnt, bb.bb_dpd_cnt, bb.bb_worst_status
    from {{ ref('stg_bureau') }} b
    left join {{ ref('int_bureau_balance_by_bureau') }} bb using (sk_id_bureau)
)

select
    sk_id_curr,
    count(*)                                                  as bureau_cnt,
    count(*) filter (where credit_active = 'Active')          as bureau_active_cnt,
    count(*) filter (where credit_active = 'Closed')          as bureau_closed_cnt,
    count(*) filter (where credit_active = 'Bad debt')        as bureau_bad_debt_cnt,
    count(distinct credit_type)                               as bureau_credit_type_nunique,
    count(*) filter (where days_credit >= -730)               as bureau_cnt_last_2y,

    min(days_credit)                                          as bureau_days_credit_min,
    max(days_credit)                                          as bureau_days_credit_max,
    avg(days_credit)                                          as bureau_days_credit_mean,
    max(days_credit_enddate)                                  as bureau_days_enddate_max,
    avg(days_credit_enddate)                                  as bureau_days_enddate_mean,

    sum(amt_credit_sum)                                       as bureau_credit_sum,
    avg(amt_credit_sum)                                       as bureau_credit_mean,
    max(amt_credit_sum)                                       as bureau_credit_max,
    sum(amt_credit_sum_debt)                                  as bureau_debt_sum,
    sum(amt_credit_sum_overdue)                               as bureau_overdue_sum,
    sum(amt_credit_sum_debt)::double precision
        / nullif(sum(amt_credit_sum), 0)                      as bureau_debt_credit_ratio,

    max(credit_day_overdue)                                   as bureau_day_overdue_max,
    count(*) filter (where credit_day_overdue > 0)            as bureau_overdue_cnt,
    max(amt_credit_max_overdue)                               as bureau_max_overdue_amt,
    sum(cnt_credit_prolong)                                   as bureau_prolong_sum,

    sum(bb_months_cnt)                                        as bureau_bb_months_sum,
    sum(bb_dpd_cnt)                                           as bureau_bb_dpd_sum,
    sum(bb_dpd_cnt)::double precision
        / nullif(sum(bb_months_cnt), 0)                       as bureau_bb_dpd_share,
    max(bb_worst_status)                                      as bureau_bb_worst_status
from joined
group by sk_id_curr