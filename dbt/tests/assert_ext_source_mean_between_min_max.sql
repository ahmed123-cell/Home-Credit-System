-- External source mean must mathematically lie between ext_source_min and ext_source_max
-- (Allowing 1e-6 floating-point rounding tolerance)
select
    sk_id_curr,
    ext_source_mean,
    ext_source_min,
    ext_source_max
from {{ ref('mart_application_features') }}
where ext_source_mean is not null
  and (ext_source_mean < (ext_source_min - 1e-6)
    or ext_source_mean > (ext_source_max + 1e-6))
