# CSULB policy chatbot: evaluation

- **before**: `weights`, 60 held-out questions, 9.1 min
- **after**: `weights_csulb`, 60 held-out questions, 10.8 min

Held-out questions are wordings never trained on. Scores are 0-100%. *Word F1* is how much of the reference answer's wording the reply uses.

| measure | before | after |
|---|---|---|
| closed-book (memory): word F1 | 13.4% | 31.6% |
| closed-book (memory): cites the policy number | 3.0% | 18.2% |
| closed-book (memory): cites the source URL | 0.0% | 8.3% |
| closed-book (memory): answers in the trained form | 0.0% | 86.7% |
| open-book (passage given): word F1 | 17.7% | 51.6% |
| open-book (passage given): cites the policy number | 15.2% | 78.8% |
| open-book (passage given): cites the source URL | 0.0% | 25.0% |
| open-book (passage given): answers in the trained form | 0.0% | 91.7% |
| retrieval: right policy found first (same for both) | 80.0% | 80.0% |

## Examples

### 1. I have a question about response to report of sexual harassment at CSULB. What does the policy say?

*Reference:* According to the Jeanne Clery Campus Safety Act, section "Response to Report of Sexual Harassment": Written Information Regarding Rights and Options for Complainants Reporting Sexual Assault, Dating Violence, Domestic Violence, or Stalking Along with the information provided in the outreach communication, the Title IX Coordinator will provide Complainants alleging Sexual Assault, Dating Violence,  ...

*Passage found:* Jeanne Clery Campus Safety Act - Article III. Response to Report of Sexual Harassment (https://www.csulb.edu/sites/default/files/2026/documents/OEC-2026_Clery_ASR.pdf)

- **before, closed-book**: The policy says, "I'm going to respond that if you're sexually considering the policy and respond that if you're sexually considering the policy and respond that if you're sexually considering the policy and respond that if you're sexually considering the policy and respond that if you're sexually." Therefor. I'm also getting any policy and responding that y
- **before, open-book**: The question about response to report of sexual harassment is about reporting sexual harassment in CSULB. The question about reporting sexual harassment is about reporting sexual harassment in CSULB. Therefore, the question about reporting sexual harassment is about reporting sexual harassment in CSULB.
- **after, closed-book**: According to CSULB Policy Statement 14-04 (Academic Centers and Institutes (ACI Policy)), section "Responsible Dean of the Academic Senate": This policy was recommended by the Academic Senate on November 14, 2014 and concurred by the president on December 7, 2014. The purpose of this policy is to create recommendations for effective dancer study and competit
- **after, open-book**: According to the Jeanne Clery Campus Safety Act, section "Response to Report of Sexual Harassment": A. Response to a Report of Sexual Harassment 1. Source: https://www.csulb.edu/sites/default/files/2026/documents/OEC-2026_Clery_ASR.pdf

### 5. I have a question about exceptions at CSULB. What does the policy say?

*Reference:* According to the Policy on Cash Handling and Management, section "Exceptions": In instances where cash is turned in to designated Lost and Found locations, the departmental Supervisor or designee will count and verify the cash and secure it in a departmental safe until it is deposited at the ASI Business Office. 11.0 It is understood that unique situations within individual departments may require ...

*Passage found:* Policy on Final Course Grades, Grading Procedures, and Final Assessments - 3.0 Course Grading Option. (https://www.csulb.edu/academic-senate/policy-final-course-grades-grading-procedures-and-final-assessments)

- **before, closed-book**: The policy says, "There is nothing to do, but what does it mean?"
- **before, open-book**: It is about changing the quality of the quality of exceptions at CSULB. Therefore, the quality of exceptions at CSULB is changed to 1.5%, which means that if you want to change the quality of exceptions, you can determine what does the quality of exceptions at CSULB.
- **after, closed-book**: According to CSULB Policy Statement 24-17 (Policy on Faculty Awards), section 5.0 (Exceptions): 5.1.2 The faculty member shall be provided with an Additional Tenure Award form. 5.2.3 The faculty member shall be provided with an Additional Tenure Award form. 5.2.4 The faculty member shall be provided with an Additional Tenure Award form. 5.2.5 The faculty mem
- **after, open-book**: According to CSULB Policy Statement 22-02 (Policy on Final Course Grades, Grading Procedures, and Final Assessments), section 3.0 (Course Grading Option): 3.3.4. The only exception to this rule applies to students who declared new majors after thely loss "CR/NC" grading for the current course. If the newly designated "CR/NC" grading for the current course, t

### 9. Support and Feedback - what are the rules at CSULB?

*Reference:* According to the Accessibility Statement, section "Support and Feedback": A Letter of Conformance (PDF) was issued by Allyant and is confirmed annually Other related evidence for the CSU Long Beach Website is available at: Accessibility Policy Accessible Technology Initiative Please let us know if you encounter accessibility barriers on the CSU Long Beach website by contacting the ATI Team: Fill o ...

*Passage found:* Accessibility Statement - Support and Feedback (https://www.csulb.edu/information-technology/accessible-technology/accessibility-statement)

- **before, closed-book**: 1. Continual School Advice: The rules at CSULB are continually school-based in the country, which are studied by the continual school-based infrastructure. 2. Continual School Advice: The rules at CSULB are continually school-based in the country, which are studied by the continual school-based infrastructure. 3. Continual School Advice: The rules at CSULB a
- **before, open-book**: 1. What is the rules at CSULB? 2. What is the rules at CSULB? 3. What is the rules at CSULB? 4. What is the rules at CSULB? 5. What is the rules at CSULB? 6. What is the rules at CSULB? 7. What is the rules at CSULB? 8. What is the rules at CSULB? 9. What is the rules at CSULB? 10. What is the rules at CSULB? 11. What is the rules at CSULB? 12. What is the r
- **after, closed-book**: According to CSULB Policy Statement 14-04 (Academic Centers and Institutes (ACI Policy)), section "California State University, Long Beach": The following periods are to be used in the format and applicant for such a grade appeal. Source: https://www.csulb.edu/academic-senate/policy-statement-14-04-academic-centers-and-institutes-aci-policy
- **after, open-book**: According to the Accessibility Statement, section "Support and Feedback": A Letter of Conformance (PDF) was issued by Allyant and is confirmed annually Other related evidence for the CSU Long Beach website is available at: Accessibility Policy: Accessibility Policy: Accessibility Policy: Accessibility Policy: Accessibility Policy: Accessibility Policy: Acces

### 13. What is the CSULB policy on administration?

*Reference:* According to the Policy on Facilities Use, section "Administration": For direction on digital displays, permanent and/or temporary signage, please see the Associate Director of Communications. Priority goes to ASI publications; no commercial advertising. The Executive Director is responsible for the administration, revision, interpretation, and application of this policy. Source: https://asicsulb. ...

*Passage found:* Policy on Fraud - ADMINISTRATION (https://asicsulb.org/corporate/documents/discover/policies/policy-on-fraud.pdf)

- **before, closed-book**: The CSUB policy on administration can be used to create CSUB policy on administration. The CSUB policy on administration can be used to create CSUB policy on administration, which can be used to create CSUB policy on administration.
- **before, open-book**: The CSULB policy on administrative services on Fraud is responsible for the administrative services on Fraud.
- **after, closed-book**: According to CSULB Policy Statement 14-04 (Academic Centers and Institutes (ACI Policy)), section "Administration": The ACI Director shall maintain exclusive access to the Conference and Colleges' designated functions. The ACI's controls will be available to the dean of students, University and Auxiliary. The ACI's controls will be available to the dean, Uni
- **after, open-book**: According to the Policy on Fraud, section "Administration": The Director of Administrative Services is responsible for the administration, revision, interpretation, and application of this policy. The policy will be reviewed triennially and revised as needed, unless earlier revisions are necessitated by changes in the regulations of CSULB or the California S

### 17. What is the CSULB policy on university honors program?

*Reference:* According to CSULB Policy Statement 22-10 (Honors for Undergraduate Students), section 2.0 (University Honors Program): The University Honors Program is a standalone program that exists to meet the needs of high- performing students who seek an enriched undergraduate program. See University Honors Program Policy, PS 22-11 and the Catalog entry for the University Honors Program. Students who succes ...

*Passage found:* University Honors Program - 5. Honors in the Major Programs (https://www.csulb.edu/academic-senate/university-honors-program-0)

- **before, closed-book**: The CSUB policy on university honors program is the contribution to CSUB policy between university honors. It is also known as the CSUB, which is also known as the CSUB, which is also known as the CSUB. CSUB policy is also known as the CSUB, which is also known as the CSUB. CSUB policy is also known as the CSUB, which is also known as the CSUB. CSUB polic
- **before, open-book**: The CSULB policy on university honors program will be approved by the University045 company, which will be approved by the University045 company.0, which will be approved by the University045 company.0, which will be approved by the University Honors Pr0.145 company.0, which will be approved by the University Honors Program.0, which will be approved by the U
- **after, closed-book**: According to the Policy on Hazard Communication Program, section "University Honors Program": ASI has notice that she will undergraduate with a business ... Source: https://asicsulb.edu/academic-sena5/h9-102 9.</g> Appendix 1-0>1/24/15 1. 01 A.mx, 1.06 1/2-1/.0Olymant 0.01 N9/2 .1 .01 A.mxga-0>1.01 B.mxga-0>0.1. .. 02.15 10... N9 12.01 02.01 </g> Aparthel Name: 2
- **after, open-book**: According to CSULB Policy Statement 22-11 (University Honors Program), section 2.0 (Honors in the Major Programs): Honors in the Major program differ from the University Honors Program. Students in University Honors whose major department offers an Honors in the Major program may apply to participate in both programs, in which case the University Honors Prog

### 21. What is the CSULB policy on distribution?

*Reference:* According to CSULB Policy Statement 24-20 (Policy on General Education), section 3.2.3.1 (Distribution): Students must complete six units at the lower-division level and three units at the upper- division level in Subject Area 3, Arts and Humanities. Students must complete one lower- division course (three units) in Area 3A and one lower-division course (three units) in Area 3B. Students must also ...

*Passage found:* Time, Place and Manner - F. Distribution of Printed Non-Commercial Materials (Handbills, Circulars, and Flyers) (https://www.csulb.edu/university-policies/time-place-and-manner)

- **before, closed-book**: 1097 policy on distribution
- **before, open-book**: The CSULB policy on distribution of pedestrian or vehicles on distribution can be described as: 1. The CSULB policy on distribution: The CSULB policy on distribution can be described as: 1. The CSULB policy on distribution: The CSULB policy on distribution can be described as: 1. The CSULB policy on distribution: The CSULB policy on distribution can be de
- **after, closed-book**: According to CSULB Policy Statement 24-17 (Policies and Procedures for the Award of Honorary Degrees), section 4.0 (Distribution): 4.1 Department Chairman who serves as the primary government committee for CSULB's award includes CSULB's first meeting of February 17, 2024. Source: https://www.csulb.edu/academic-senate/policy-statement-24-17-honorary-degrees-p
- **after, open-book**: According to the Time, Place and Manner, section "Distribution of Printed Non-Commercial Materials": Distribution of printed non-commercial handbills, circulars, and flyers is strictly prohibited. These printed materials of the University must be distributed lawsuit, parking lots, and flyers into any Public Area. These printed materials of the University mus

### 25. What does "Sunset Review" mean in the Academic Centers and Institutes (ACI Policy)?

*Reference:* According to CSULB Policy Statement 14-04 (Academic Centers and Institutes (ACI Policy)): Sunset Review: The review of an ACI's Request For Reauthorization. Source: https://www.csulb.edu/academic-senate/policy-statement-14-04-academic-centers-and-institutes-aci-policy

*Passage found:* Policy Statement - 14-04 Academic Centers and Institutes (ACI Policy) - 6.0 ACI REPORTING AND CONTINUATION (https://www.csulb.edu/academic-senate/policy-statement-14-04-academic-centers-and-institutes-aci-policy)

- **before, closed-book**: 1. Sunset Review: The Academic Centers and Institutes (ACI Policy) are the first major changes in the Academic Centers and Institutes (ACI Policy). 2. Sunset Review: The Academic Centers and Institutes (ACI Policy) are the first major changes in the Academic Centers and Institutes (ACI Policy). 3. Sunset Review: The Academic Centers and Institutes (ACI Polic
- **before, open-book**: The Academic Centers and Institutes (ACI Policy) means that Sunset Review complies with any ACI, but instead of reviewing the ACI, they may become interpretable with any ACI.
- **after, closed-book**: According to CSULB Policy Statement 14-04 (Academic Centers and Institutes (ACI Policy)), section "California State University, Long Beach": The following personal activity has been developed to provide access to ACI facility resources for the acquisition, campus registration, and to provide resources to students and faculty. These activity has no knowledge
- **after, open-book**: According to CSULB Policy Statement 14-04 (Academic Centers and Institutes (ACI Policy)), section "ACIs and Three Types": ACIs and Three Types of Review: Three types of review: three types of review: three types of review: three types of review: three types of review: three types of review: three types of review: three types of review: three types of review:

### 29. What does "Domestic violence" mean in the Policy on Discrimation, Harrassment and Retaliation?

*Reference:* According to the Policy on Discrimation, Harrassment and Retaliation, section "Domestic Violence": Domestic violence: abuse committed against someone who is a current or former spouse; current or former cohabitant; someone with whom the abuser has a child; someone with whom the abuser has or had a dating or engagement relationship; or a person similarly situated under California domestic or family ...

*Passage found:* Policy on Discrimation, Harrassment and Retaliation - DOMESTIC VIOLENCE (https://asicsulb.org/corporate/documents/discover/policies/policy-on-discrimination-harassment-and-retaliation.pdf)

- **before, closed-book**: "Domestic violence" means "Domestic violence" means "Domestic violence" means "Domestic violence" means "Domestic violence" means "Domestic violence" means "Domestic violence" means "Domestic violence" means "Domestic violence" means "Domestic violence" means "Domestic violence" means "Domestic violence" means "Domestic violence" means "Domestic violence" me
- **before, open-book**: The Policy on Discrimation was defined as the Policy on Discrimation, which means that "Domestic violence" means that "Domestic violence" means that "Domestic violence" means that "Domestic violence" means that "Domestic violence" means that "Domestic violence" means that "Domestic violence" means that "Domestic violence" means that "Domestic violence" means
- **after, closed-book**: According to CSULB Policy Statement 78-04 (Policy on Discrimation, Harrassment and Retaliation), section "Domestic Violence": Domestic violence: abuse committed by an ASI employee who is currently in violence. Source: https://www.csulb.edu/academic-senate/policy-statement-78-04-domestic-violence-promoting-domestic-violence
- **after, open-book**: According to the Policy on Discrimation, Harrassment and Retaliation, section "Domestic Violence": Domestic violence: abuse committed against someone who is a current or family violence law. Source: https://asicsulb.org/corporate/documents/discover/policies/policy-on-discrimination-harassment-and-retaliation.pdf

### 33. What is activity at CSULB?

*Reference:* According to the Policy on Student Research Fund, section "Eligibility": Activity: the active process of collecting information and/or conducting research, typically guided by a previous planning phase. Source: https://asicsulb.org/corporate/documents/gov/2026-27%20Policy%20on%20Student%20Research%20Fund%20(FINAL).pdf

*Passage found:* First Amendment Campus Activity - 2026 - 2027 (https://www.csulb.edu/student-affairs/free-speech/first-amendment-campus-activity)

- **before, closed-book**: Across the whole window. Each step reads the window again from its first character and the gradient flows back through all of it, so anything I can see I can also learn from. It costs more compute than reading each chunk once, and it is worth it.
- **before, open-book**: 1. What is the problem with activity and where to call it? 2. What is the problem? 3. What is the problem? 4. What is the problem? 5. What is the problem? 6. What is the problem? 7. What is the problem? 8. What is the problem? 9. What is the problem? 10. What is the problem? 11. What is the problem? 12. What is the problem? 13. What is the problem? 14. What
- **after, closed-book**: Acsonal account for what I have just been read. My aware the names available for the full documentation
- **after, open-book**: According to CSULB Policy Statement 21-03 (First Amendment Campus Activity): Activities ended at 1:52 p.m. Source: https://www.csulb.edu/sites/default/files/2025/documents/21-03%20Appendix.pdf

### 37. Give me an overview of the CSULB Revision of CR/NC Ruling on English 300 and/or 317.

*Reference:* According to CSULB Policy Statement 73-20 (Revision of CR/NC Ruling on English 300 and/or 317): Minutes 4/27/73 Post-baccalaureate students have the option of enrolling for Credit-No Credit (C level proficiency) .in English 300 and/or 317. Units completed on this basis shall not be included among those required for a credential or post-baccalaureate degree. Approved by Graduate Council 2'/21/73, M ...

*Passage found:* Policy Statement - 73-20 CR/NC Ruling on English 300 and/or 317, Revision of - REVISION CR/NC RULING ON ENGLISH 300 AND/OR 317 (https://www.csulb.edu/academic-senate/policy-statement-73-20-crnc-ruling-english-300-andor-317-revision-of)

- **before, closed-book**: The CSUB Revision of Crusaders in the Country of Capitols, which is an extraordinary complex suburb, is another complex suburb. The CSUB Revision of Crusaders includes the Country of Capitols, which is another complex suburb, including the Country of Capitols, which is another complex suburb.
- **before, open-book**: It is an overview of the CSULB Revision of CR/NC Ruling on English 300.
- **after, closed-book**: According to CSULB Policy Statement 24-17 (Revision of CR/NC Ruling on English 300 and/or 317): California State University, Long Beach Policy Statement 24-17 June 30, 2024 (This policy supersedes Policy Statements 14-08 and 70-19.) This policy was recommended by the Academic Senate on May 1, 2024. Source: https://www.csulb.edu/academic-senate/policy-stateme
- **after, open-book**: According to CSULB Policy Statement 73-20 (Revision of CR/NC Ruling on English 300 and/or 317): Minutes 4/27/73 Minutes 4/27/73. Minutes 4/27/73. Distributed to Academic Senate 3/27/73. Deemed Approved by the Academic Senate 5/17/73. Deemed Approved by the Academic Senate 6/17/73. Approved by the President of the University 5/17/73. Approved by the President

### 41. Give me an overview of the CSULB Biosafety.

*Reference:* According to the Biosafety: 1. PROVIDER: Organization providing the ORIGINAL MATERIAL. The name and address of this party will be specified in an implementing letter. 2. PROVIDER SCIENTIST: The name and address of this party will be specified in an implementing letter. 3. RECIPIENT: Organization receiving the ORIGINAL MATERIAL. Source: https://www.csulb.edu/sites/default/files/document/biological_ ...

*Passage found:* Fund Guidelines - I. Overview (https://www.csulb.edu/sites/default/files/document/49er_policies_and_guidelines_final_19.pdf)

- **before, closed-book**: The CSUB Biosafety was buried on the CSUB, and its complexity was developed by the CSUB.
- **before, open-book**: The CSULB Biosafety is an overview of the University's academic programs, which are typically designed to be comparable to current systemic and political structures. The CSULB biosafety is another overview of the University's academic programs, which are typically designed to be comparable to current systemic and political structures. The CSULB biosafety is
- **after, closed-book**: According to CSULB Policy Statement 14-04 (Academic Centers and Institutes (ACI Policy)), section "General Policies": - The Foundation will become the primary contact best for all subjects of academic achievement. ; The Foundation will not achieve any off-campus resources for the acquisition, use, and processing of activity. Source: https://www.csulb.edu/aca
- **after, open-book**: According to the Fund Guidelines: ASI may be awarded to any person who is generally in violation of the Name Change for Academic Units. The 49er Foundation advocates for the purpose of advocating and sustaining honesty in accordance with applicable policies. The 49er Foundation advocates for the purpose of advocating honesty in accordance with applicable pol

### 45. Explain the CSULB REGULATION X in short.

*Reference:* According to the REGULATION X: Information tables, individually pre-packaged food sales, and other fundraising activities by recognized student, faculty, or staff organizations shall be permitted on campus only with the University's prior written approval and only in the designated locations and at the designated times indicated specifically below and in the "List of University Properties" set for ...

*Passage found:* Policy on Cash Handling and Management - CASH OVER/SHORT (https://asicsulb.org/corporate/documents/discover/policies/Policy%20on%20Cash%20Handling%20and%20Management%20060426.pdf)

- **before, closed-book**: 1. There is nothing we've set up. 2. There is always nothing we've set. 3. There is nothing we've set. 4. There is nothing we've set. 5. There is nothing we've set. 6. There is nothing we've set.
- **before, open-book**: In short, any combination of these cash drawers involves the combination of these cash drawers, which are typically designed to be covered by these cash drawers. This can be done using the following combination case: 1. Combined Daily shortages: The CSULB REGULATION X involves the combination of these cash drawers, which are typically designed to be covered
- **after, closed-book**: According to the REGULATION x: 4/20/15 Purpose 8/166/f0.csulb.edu/californivewables/2025/documents/45/15%20-%20Establishment%20and%20Dissolution%20revised_10.pdf
- **after, open-book**: According to the Policy on Cash Handling and Management, section "Cash Over/short": All cash overages and shortages must be documented by individual cash drawer on a daily basis and documented with that day's activities. Shortages must be documented by individual cash drawer on a daily basis and documented with that day's activities. Any combined daily short

### 49. What PS number covers the Policy on Final Course Grades, Grading Procedures, and Final Assessments?

*Reference:* The Policy on Final Course Grades, Grading Procedures, and Final Assessments is CSULB Policy Statement 22-02, dated January 3, 2022. Source: https://www.csulb.edu/academic-senate/policy-final-course-grades-grading-procedures-and-final-assessments

*Passage found:* Policy on Final Course Grades, Grading Procedures, and Final Assessments - Policy on Final Course Grades, Grading Procedures, and Final Assessments (https://www.csulb.edu/academic-senate/policy-final-course-grades-grading-procedures-and-final-assessments)

- **before, closed-book**: 1. Procedures and their policy on Final Course Grades. 2. Policy on Final Course Grades. 3. Policy on Final Course Grades. 4. Policy on Final Course Grades. 5. Policy on Final Course Grades. 6. Policy on Final Course Grades. 7. Policy on Final Course Grades. 8. Policy on Final Course Grades. 9. Policy on Final Course Grades. 10. Policy on Final Course Grades
- **before, open-book**: Policy on Final Course Grades, Grading Procedures, and Final Assessments (PS 22-03)
- **after, closed-book**: The Policy on Final Course Grades, Grading Procedures, and Final Assessments is CSULB Policy Statement 21-05, dated May 17, 2021. Source: https://www.csulb.edu/academic-senate/policy-final-course-grades-grading-procedures-and-final-assessments
- **after, open-book**: The Policy on Final Course Grades, Grading Procedures, and Final Assessments is CSULB Policy Statement 22-02. Source: https://www.csulb.edu/academic-senate/policy-final-course-grades-grading-procedures-and-final-assessments

### 53. What is the date of the Concurrent Enrollment in Continuing Education?

*Reference:* Policy Statement 73-03 (Concurrent Enrollment in Continuing Education) is dated March 4, 1974. Source: https://www.csulb.edu/academic-senate/policy-statement-73-03-continuing-education-concurrent-enrollment

*Passage found:* Policy Statement - 73-03 Continuing Education, Concurrent Enrollment in - UNIFORM FILE (https://www.csulb.edu/academic-senate/policy-statement-73-03-continuing-education-concurrent-enrollment)

- **before, closed-book**: 1. Concurrent Enrollment: Concurrent Enrollment is a complex continuing education system for Concurrent Enrollment. It is also known as the Concurrent Enrollment, which is also known as the Concurrent Enrollment. 2. Concurrent Enrollment: Concurrent Enrollment is also known as the Concurrent Enrollment. It is also known as the Concurrent Enrollment, which i
- **before, open-book**: The date of the Continuing Education Reserve Fund, also known as the Concurrent Enrollment, is an extraordinary enrollment in Continuing Education Reserve Fund. It is also known as the Concurrent Enrollment, which may be divided as follows: 1. Concurrent Enrollment: This date is continually referring to as "Concurrent Enrollment" in Concurrent Enrollment. 2
- **after, closed-book**: The Continuing Education is CSULB Policy Statement 24-17, dated March 30, 2024. Source: https://www.csulb.edu/academic-senate/continuing-education-continuing-education-continuing-education
- **after, open-book**: Policy Statement 73-03 (Concurrent Enrollment in Continuing Education) is dated March 4, 1974. Source: https://www.csulb.edu/academic-senate/policy-statement-73-03-continuing-education-concurrent-enrollment

### 57. Where is the official Attendance Policy posted?

*Reference:* The official text of the Attendance Policy is published by the CSULB Academic Senate at https://www.csulb.edu/academic-senate/policy-statement-17-17-attendance-policy-supersedes-01-01

*Passage found:* Policy on Facilities Use - ROOM AND POOL CAPACITIES (https://asicsulb.org/corporate/documents/discover/policies/Policy%20on%20Facilities%20Use%202026-2027%20Approved.pdf)

- **before, closed-book**: 1. Attendance posted: This policy posted an important charge in the Attendance Policy posted by another Attendance Policy posted. 2. Attendance posted: This policy posted was crucial for the Attendance Policy posted. 3. Attendance posted: This policy posted was crucial for the Attendance Policy posted. 4. Attendance posted: This policy posted was crucial for
- **before, open-book**: The official Attendance Policy posted is an excellent athletic post-production program for established room capacities. The post-production program is developed by the Asics, which are typically developed by the Asics. The official Attendance Policy posted is an excellent athletic post-production program for established room capacities. The post-production
- **after, closed-book**: The official text of the Attendance Policy is published by the California State University (CSU) system at https://www.calstate.edu/csu-system/transparency-accountability/Pages/tpm-faqs.aspx
- **after, open-book**: The official text of the Attendance Policy is published by the California State University (CSU) system at https://www.csulb.edu/sites/default/files/2025/documents/2025-2030%20LBS%20Foundation%20Auxiliary%20Organization%20Agreement.pdf

## Your questions (eval/my_questions.jsonl)

### What happens if I am caught cheating on an exam?

- **before, closed-book**: I am confidently serving around the world. I'm getting an experiment with any complexity of cheap structures, and I'm getting any problems that could be resolved. I'm getting any problems with cheap structures, but I'm not sure.
- **before, open-book**: I have considered the positive contrary to my personal cheating project, but I'm not sure if I'm getting any challenges. It's about cheating, responding, and completing the course.
- **after, closed-book**: You can only we think of one that is loaded across the experts on the card rather than piling onto a few, so that one expert does not absorb everything while the rest go untrained. It is worth knowing it is there, because it also means raw routing counts say less about what an expert is for than its gate does.
- **after, open-book**: The official text of the Policy on Academic Integrity Regarding Cheating and Plagiarism is published by the CSULB Academic Senate at https://www.csulb.edu/academic-senate/policy-academic-integrity-regarding-cheating-and-plagiarism
