select
    sk_id_curr::bigint as sk_id_curr,
    sk_id_bureau::bigint as sk_id_bureau,
    credit_active as credit_active,
    credit_currency as credit_currency,
    days_credit::bigint as days_credit,
    credit_day_overdue::bigint as credit_day_overdue,
    days_credit_enddate::double precision as days_credit_enddate,
    days_enddate_fact::double precision as days_enddate_fact,
    amt_credit_max_overdue::double precision as amt_credit_max_overdue,
    cnt_credit_prolong::bigint as cnt_credit_prolong,
    amt_credit_sum::double precision as amt_credit_sum,
    amt_credit_sum_debt::double precision as amt_credit_sum_debt,
    amt_credit_sum_limit::double precision as amt_credit_sum_limit,
    amt_credit_sum_overdue::double precision as amt_credit_sum_overdue,
    credit_type as credit_type,
    days_credit_update::bigint as days_credit_update,
    amt_annuity::double precision as amt_annuity
from {{ source('raw', 'bureau') }}
