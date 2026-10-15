-- All applicants must be of legal adult age (between 18 and 100 years old)
select
    sk_id_curr,
    age_years
from {{ ref('mart_application_features') }}
where age_years < 18 or age_years > 100
