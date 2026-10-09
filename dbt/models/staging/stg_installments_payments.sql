select
    sk_id_prev::bigint as sk_id_prev,
    sk_id_curr::bigint as sk_id_curr,
    num_instalment_version::double precision as num_instalment_version,
    num_instalment_number::bigint as num_instalment_number,
    days_instalment::double precision as days_instalment,
    days_entry_payment::double precision as days_entry_payment,
    amt_instalment::double precision as amt_instalment,
    amt_payment::double precision as amt_payment
from {{ source('raw', 'installments_payments') }}
