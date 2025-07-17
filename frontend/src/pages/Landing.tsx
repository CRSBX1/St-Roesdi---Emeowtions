import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowRight, Check } from "lucide-react";
import { useNavigate } from "react-router-dom";
import Navigation from "@/components/shared/Navigation";
import Footer from "@/components/shared/Footer";
import TestimonialCarousel from "@/components/shared/TestimonialCarousel";
import heroImage from "@/assets/cat-alone-hero.png";

const Landing = () => {
  const navigate = useNavigate();

  const plans = [
    {
      name: "Free",
      price: "$0",
      period: "/month",
      uses: "5 uses per month",
      features: ["Basic emotion analysis", "Standard recommendations", "Pet profile management", "Limited to 2 pet registrations", "Pet emotion history only up to 1 month"],
      popular: false
    },
    {
      name: "Premium",
      price: "$9.99", 
      period: "/month",
      uses: "150 uses per month (immediate), Additional 5 daily uses per day",
      features: ["Up to 300 uses per month", "Advanced emotion analysis", "Personalized recommendations", "Unlimited pet registrations","Detailed emotion history stored up to 3 months", "Priority support", "Video analysis", "Ad free"],
      popular: true
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      {/* Hero Section */}
      <section className="relative py-20 px-4 overflow-hidden bg-gradient-to-br from-background via-secondary/20 to-accent/20">
        <div className="container mx-auto max-w-6xl">
          <div className="relative">
            {/* Overlapping design */}
            <div className="grid lg:grid-cols-2 gap-8 items-center">
              <div className="text-center lg:text-left space-y-6 z-10 relative">
                <h1 className="text-5xl lg:text-7xl font-bold text-foreground drop-shadow-sm">
                  Emeowtions
                </h1>
                <p className="text-xl lg:text-2xl text-muted-foreground">
                  Emotion analysis made purr-sonal.
                </p>
                <Button 
                  size="lg" 
                  className="bg-gradient-to-r from-primary to-accent hover:from-primary-hover hover:to-accent-hover text-primary-foreground px-8 py-6 text-lg shadow-lg"
                  onClick={() => navigate('/home')}
                >
                  Start now <ArrowRight className="ml-2 h-5 w-5" />
                </Button>
              </div>
              <div className="flex justify-center lg:justify-end">
                <img 
                  src={heroImage} 
                  alt="Cute cat representing emotion analysis" 
                  className="w-80 h-80 lg:w-96 lg:h-96 object-contain"
                />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Testimonials Carousel */}
      <TestimonialCarousel />

      {/* Introduction */}
      <section className="py-16 px-4 bg-gradient-to-br from-accent/10 to-success/10">
        <div className="container mx-auto max-w-4xl text-center space-y-8">
          <h2 className="text-4xl font-bold text-foreground">Understanding Your Cat's Emotions</h2>
          <p className="text-lg text-muted-foreground leading-relaxed">
            Emeowtions uses advanced AI technology to analyze your cat's vocalizations, body language, and behavior patterns 
            to determine their emotional state. Whether your feline friend is happy, stressed, playful, or needs medical attention, 
            our platform provides personalized recommendations to help you respond appropriately and strengthen your bond with your pet.
          </p>
          <div className="grid md:grid-cols-3 gap-6 mt-12">
            <div className="text-center space-y-4 p-6 rounded-xl bg-card shadow-lg hover:shadow-xl transition-shadow">
              <div className="w-16 h-16 bg-gradient-to-br from-accent to-accent-hover rounded-full flex items-center justify-center mx-auto shadow-md">
                <span className="text-2xl">🎤</span>
              </div>
              <h3 className="text-xl font-semibold">Voice Analysis</h3>
              <p className="text-muted-foreground">Upload audio or record live to analyze meows, purrs, and other vocalizations.</p>
            </div>
            <div className="text-center space-y-4 p-6 rounded-xl bg-card shadow-lg hover:shadow-xl transition-shadow">
              <div className="w-16 h-16 bg-gradient-to-br from-success to-success-hover rounded-full flex items-center justify-center mx-auto shadow-md">
                <span className="text-2xl">🤖</span>
              </div>
              <h3 className="text-xl font-semibold">AI Insights</h3>
              <p className="text-muted-foreground">Get instant emotion detection with personalized care recommendations.</p>
            </div>
            <div className="text-center space-y-4 p-6 rounded-xl bg-card shadow-lg hover:shadow-xl transition-shadow">
              <div className="w-16 h-16 bg-gradient-to-br from-lavender to-lavender-hover rounded-full flex items-center justify-center mx-auto shadow-md">
                <span className="text-2xl">📊</span>
              </div>
              <h3 className="text-xl font-semibold">Track Progress</h3>
              <p className="text-muted-foreground">Monitor your cat's emotional wellbeing over time with detailed history.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Pricing Plans */}
      <section className="py-16 px-4 bg-gradient-to-br from-lavender/20 to-primary/10">
        <div className="container mx-auto max-w-4xl">
          <h2 className="text-4xl font-bold text-center mb-12">Choose Your Plan</h2>
          <div className="grid md:grid-cols-2 gap-8 max-w-3xl mx-auto">
            {plans.map((plan, index) => (
              <Card key={index} className={`relative ${plan.popular ? 'ring-2 ring-primary shadow-xl bg-gradient-to-br from-card to-secondary/30' : 'bg-card shadow-lg hover:shadow-xl'} transition-all duration-300`}>
                {plan.popular && (
                  <Badge className="absolute -top-3 left-1/2 transform -translate-x-1/2 bg-gradient-to-r from-primary to-accent text-primary-foreground">
                    Most Popular
                  </Badge>
                )}
                <CardContent className="p-8 text-center space-y-6">
                  <div>
                    <h3 className="text-2xl font-bold text-foreground">{plan.name}</h3>
                    <div className="mt-2">
                      <span className="text-4xl font-bold text-foreground">{plan.price}</span>
                      <span className="text-muted-foreground">{plan.period}</span>
                    </div>
                    <p className="text-lg text-muted-foreground mt-2">{plan.uses}</p>
                  </div>
                  <ul className="space-y-3">
                    {plan.features.map((feature, i) => (
                      <li key={i} className="flex items-center justify-center gap-2">
                        <Check className="h-4 w-4 text-success" />
                        <span className="text-sm">{feature}</span>
                      </li>
                    ))}
                  </ul>
                  <Button 
                    className={`w-full ${plan.popular ? 'bg-gradient-to-r from-primary to-accent hover:from-primary-hover hover:to-accent-hover' : 'bg-secondary hover:bg-secondary-hover'} transition-all duration-300`}
                    onClick={() => navigate(plan.popular ? '/billing' : '/home')}
                  >
                    {plan.popular ? 'Upgrade to Premium' : 'Get Started'}
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Landing;