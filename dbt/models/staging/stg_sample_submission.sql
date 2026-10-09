select
    sk_id_curr::bigint as sk_id_curr,
    target::double precision as target
from {{ source('raw', 'sample_submission') }}
