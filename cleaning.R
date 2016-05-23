library("dplyr")

target_users <- read.csv("target_users.csv")

interactions <- read.csv("interactions-v01.csv")
interactions_per_item <- interactions %>%
  filter(interaction_type != 4) %>%
  group_by(item_id) %>%
  summarize(total = n())

top_item_interactions <- top_n(interactions_per_item, 20, total) %>%
  arrange(desc(total))

#TARGET USERS QUE NAO INTERAGIRAM
interactions_per_user <- interactions %>%
  select(user_id) %>%
  distinct()

users_winteracion <- target_users %>%
  filter(!user_id %in% interactions_per_user$user_id)

#TRAIN USERS
interactions <- read.csv("interactions-v01.csv")
train_interactions <- interactions %>% filter(week == 45)

  
write.table(users_winteracion, "users_wo_interactions.csv", 
            row.names = FALSE, col.names=FALSE, sep=",")

write.table(top_item_interactions, "top_items.csv",
            row.names = FALSE, col.names=FALSE, sep=",")

write.table(train_interactions, "train_interactions.csv",
            row.names = FALSE, col.names=FALSE, sep=",")
