select
    sk_id_prev::bigint as sk_id_prev,
    sk_id_curr::bigint as sk_id_curr,
    months_balance::bigint as months_balance,
    cnt_instalment::double precision as cnt_instalment,
    cnt_instalment_future::double precision as cnt_instalment_future,
    name_contract_status as name_contract_status,
    sk_dpd::bigint as sk_dpd,
    sk_dpd_def::bigint as sk_dpd_def
from {{ source('raw', 'pos_cash_balance') }}
