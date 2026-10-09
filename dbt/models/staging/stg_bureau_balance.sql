select
    sk_id_bureau::bigint as sk_id_bureau,
    months_balance::bigint as months_balance,
    status as status
from {{ source('raw', 'bureau_balance') }}
