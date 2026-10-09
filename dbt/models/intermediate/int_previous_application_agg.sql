select
    sk_id_curr,
    count(*)                                                       as prev_cnt,
    count(*) filter (where name_contract_status = 'Approved')      as prev_approved_cnt,
    count(*) filter (where name_contract_status = 'Refused')       as prev_refused_cnt,
    count(*) filter (where name_contract_status = 'Canceled')      as prev_canceled_cnt,
    count(*) filter (where name_contract_status = 'Refused')::double precision
        / count(*)                                                 as prev_refused_share,
    count(*) filter (where days_decision >= -365)                  as prev_cnt_last_1y,

    avg(amt_application)                                           as prev_amt_application_mean,
    max(amt_application)                                           as prev_amt_application_max,
    avg(amt_credit)                                                as prev_amt_credit_mean,
    max(amt_credit)                                                as prev_amt_credit_max,
    avg(amt_annuity)                                               as prev_amt_annuity_mean,
    avg(amt_down_payment)                                          as prev_amt_down_payment_mean,
    avg(amt_credit::double precision / nullif(amt_application, 0)) as prev_credit_application_ratio_mean,
    avg(cnt_payment)                                               as prev_cnt_payment_mean,

    max(days_decision)                                             as prev_days_decision_max,
    min(days_decision)                                             as prev_days_decision_min,

    -- approved loans only
    avg(amt_credit) filter (where name_contract_status = 'Approved') as prev_approved_amt_credit_mean,
    avg(amt_annuity) filter (where name_contract_status = 'Approved') as prev_approved_amt_annuity_mean
from {{ ref('stg_previous_application') }}
group by sk_id_curr