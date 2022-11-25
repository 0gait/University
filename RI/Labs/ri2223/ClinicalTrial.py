class ClinicalTrial:

    def __init__(self, nct_id, brief_title, detailed_description,
                 brief_summary, inclusion_criteria, exclusion_criteria, phase, study_type, study_design,
                 condition, intervention_name, intervention_type, gender, minimum_age, maximum_age,
                 healthy_volunteers, mesh_term):
        self.nct_id = nct_id
        self.brief_title = brief_title
        self.detailed_description = detailed_description
        self.brief_summary = brief_summary
        self.inclusion_criteria = inclusion_criteria
        self.exclusion_criteria = exclusion_criteria
        self.phase = phase
        self.study_type = study_type
        self.study_design = study_design
        self.condition = condition
        self.intervention_name = intervention_name
        self.intervention_type = intervention_type
        self.gender = gender
        self.minimum_age = minimum_age
        self.maximum_age = maximum_age
        self.healthy_volunteers = healthy_volunteers
        self.mesh_term = mesh_term
