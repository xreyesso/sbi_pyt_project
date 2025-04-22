# Load libraries
library(ggplot2)
library(dplyr)

# Read the csv file
data <- read.csv("evaluation_results.csv", header=TRUE, sep=",", na.strings="-")

# Data pre-processing
# Convert Hit to a factor
data$Hit <- factor(data$Hit, levels = c(0, 1))

# Obtain length values (removing "AA" and ",")
data$Length_numeric <- as.numeric(gsub(",", "", gsub(" AA", "", data$Length)))
summary(data)

# Plot
ggplot(data, aes(x=Length_numeric, y=as.numeric(Hit) - 1)) + 
  geom_point(alpha=0.6) +
  geom_smooth(method="glm", method.args=list(family="binomial"), se=TRUE) +
  labs(title="Probability of Pocket Detection versus Protein Length",
       x="Protein Length (AA)",
       y="Success Probability") +
  theme_minimal() +  # This replaces theme_minimal() with a cleaner theme
  theme(
    axis.line = element_line(color="black"), # Adds black axis lines
    plot.title = element_text(size=12, face="bold", hjust=0.5) # Centers and bolds title
  )

# Logistic regression with Hit as a factor
length_model <- glm(Hit ~ Length_numeric, data=data, family=binomial)
summary(length_model)

# Family analysis
# Extract family information (everything before "family" in the Protein Families column)
data$MainFamily <- gsub(" family*", "", data$Protein_Families)

# Replace empty strings with NA
data$MainFamily[data$MainFamily == ""] <- NA
data$Hit <- as.numeric(as.character(data$Hit))

# Obtain number of hits for each family
family_hits <- data %>% 
  group_by(MainFamily) %>%
  summarize(
    Total = n(),
    Hits = sum(Hit, na.rm=TRUE),
    HitRate = Hits/Total
  ) %>%
  arrange(desc(HitRate))

# Only include families with at least 2 examples and filter out NA values
family_hits_filtered <- family_hits %>% 
  filter(!is.na(MainFamily) & Total >= 2)

# Plot hit rates by protein family
ggplot(family_hits_filtered, aes(x=reorder(MainFamily, HitRate), y=HitRate)) +
  geom_bar(stat="identity", fill="steelblue") +
  geom_text(aes(label=paste0(Hits, "/", Total)), hjust=-0.1) +
  coord_flip() +
  labs(title="Pocket Detection Hit Rate by Protein Family",
       x="Protein Family",
       y="Hit Rate") +
  theme_classic() +  # This replaces theme_minimal() with a cleaner theme
  theme(
    axis.line = element_line(color="black"), # Adds black axis lines
    plot.title = element_text(size=12, face="bold", hjust=0.5) # Centers and bolds title
  )
  ylim(0, 1)
