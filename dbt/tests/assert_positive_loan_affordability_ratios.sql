-- Financial affordability ratios must be non-negative when present
select
    sk_id_curr,
    credit_income_ratio,
    annuity_income_ratio,
    credit_term,
    credit_goods_ratio
from {{ ref('mart_application_features') }}
where credit_income_ratio < 0
   or annuity_income_ratio < 0
   or credit_term < 0
   or credit_goods_ratio < 0
