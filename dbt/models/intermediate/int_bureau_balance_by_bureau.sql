{{ config(
    materialized='incremental',
    unique_key='sk_id_bureau'
) }}

select
    sk_id_bureau,
    count(*)                                              as bb_months_cnt,
    min(months_balance)                                   as bb_months_min,
    count(*) filter (where status = 'C')                  as bb_closed_cnt,
    count(*) filter (where status = 'X')                  as bb_unknown_cnt,
    count(*) filter (where status ~ '^[1-5]$')            as bb_dpd_cnt,
    max(case when status ~ '^[0-5]$' then status::int end) as bb_worst_status
from {{ ref('stg_bureau_balance') }}
group by sk_id_bureau