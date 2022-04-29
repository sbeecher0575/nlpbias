

setwd("~/College/2022aspr/ling539/final")
d <- read.csv('bias_results.csv')


library(tidyverse)

# BIAS SCORE

model_scores <- d %>% 
   group_by(filename, category, qtype, model) %>% 
   summarize(chisq = mean(model_score)) %>% 
   mutate(filename = fct_relevel(filename, sort), 
          filename = fct_relevel(filename, "glove.6B.50d"),
          filename = fct_relevel(filename, "text_all", after=Inf),
          filename = fct_rev(filename),
          source = filename,
          source = recode(source, 
                         glove.840B.300d  = '840B.300d', 
                         glove.6B.50d = '6B.50d',
                         text_all = 'All', 
                         glove.6B.100d = '6B.100d', 
                         glove.6B.200d = '6B.200d', 
                         glove.6B.300d = '6B.300d', 
                         text_acad = 'Academia', 
                         text_blog = 'Blog', 
                         text_fic = 'Fiction', 
                         text_mag = 'Magazine', 
                         text_news = 'News', 
                         text_spok = 'Spoken', 
                         text_tvm = 'TV/Movies', 
                         text_web = 'Web')
   )

model_scores %>% 
   filter(qtype == "emotion") %>% 
   ggplot(aes(x = chisq, y = source, fill = category)) +
   geom_col(position = 'dodge') + 
   geom_vline(aes(xintercept = qchisq(.95, 3), color = 'alpha = 0.05')) +
   labs(title = 'Model Scores', y = '', x=bquote(chi^2)) +
   scale_fill_discrete(name="bias category", breaks=c("race", "gender"), labels=c("race", "gender")) +
   scale_color_manual(name = "chi-square cutoff", values = c('alpha = 0.05' = 'black')) + 
   facet_grid(rows = vars(model), scales = "free_y", space="free")
      
      
model_scores %>% 
   filter(qtype == "emotion", model == 'vector') %>% 
   ggplot(aes(x = chisq, y = source, fill = category)) +
   geom_col(position = "dodge") + 
   geom_vline(aes(xintercept = qchisq(.95, 3), color = 'alpha = 0.05')) +
   labs(title = 'Vector Model Scores', y = '', x=bquote(chi^2)) +
   scale_fill_discrete(name="bias category", breaks=c("race", "gender"), labels=c("race", "gender")) +
   scale_color_manual(name = "chi-square cutoff", values = c('alpha = 0.05' = 'black'))






model_scores %>% 
   filter(qtype == "emotion", filename %in% c('glove.840B.300d', 'text_all')) %>% 
   ggplot(aes(x = category, y = chisq, fill = filename)) +
   geom_col(position = "dodge") + 
   geom_hline(yintercept = qchisq(.95, 3))

model_scores %>% 
   filter(qtype == "person",) %>% 
   ggplot(aes(x = category, y = chisq, fill = filename)) +
   geom_col(position = position_dodge(),) + 
   geom_hline(yintercept = qchisq(.95, 1))

# RACE ILLUSTRATION

race <- d %>% 
   filter(qtype == "emotion", value %in% c('African-American', 'European', 'male', 'female'), filename %in% c("text_all", "glove.840B.300d"), category == 'race') %>% 
   group_by(value, emotion, category, filename) %>% 
   summarize(weighted_count = mean(count_score))

race %>% 
   ggplot(aes(x = emotion, y = weighted_count, fill = value)) +
   geom_col(position = "dodge") +
   facet_grid(rows = vars(filename), scale="free")


# GENDER ILLUSTRATION

gender <- d %>% 
   filter(qtype == "emotion", value %in% c('female', 'male'), category == 'gender', filename == "text_acad") %>% 
   group_by(value, emotion) %>% 
   summarize(weighted_count = mean(count_score))

gender %>% 
   ggplot(aes(x = emotion, y = weighted_count, fill = value)) +
   geom_col(position = "dodge")


# ERROR ANALYSIS

ans <- read.csv('answers/text_all_emotion.csv')
ind <- sample(1:nrow(ans), size=50)
samp <- ans[ind,]
