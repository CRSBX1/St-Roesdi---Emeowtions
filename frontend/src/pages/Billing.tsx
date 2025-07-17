import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Check, ArrowLeft } from "lucide-react";
import { useNavigate } from "react-router-dom";
import Navigation from "@/components/shared/Navigation";
import Footer from "@/components/shared/Footer";

const Billing = () => {
  const navigate = useNavigate();

  const premiumFeatures = [
    "Advanced emotion analysis",
    "Personalized recommendations", 
    "Detailed emotion history",
    "Priority support",
    "Video analysis",
    "Unlimited uses"
  ];

  const handleUpgrade = () => {
    // This is where Stripe integration would go
    console.log("Upgrading to Premium...");
    // For now, just show a placeholder
    alert("Stripe integration coming soon!");
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <section className="py-20 px-4">
        <div className="container mx-auto max-w-4xl">
          <div className="flex items-center gap-4 mb-8">
            <Button
              variant="ghost"
              onClick={() => navigate(-1)}
              className="flex items-center gap-2"
            >
              <ArrowLeft className="h-4 w-4" />
              Back
            </Button>
          </div>

          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold text-foreground mb-4">
              Upgrade to Premium
            </h1>
            <p className="text-xl text-muted-foreground">
              Unlock advanced features for better cat emotion analysis
            </p>
          </div>

          <div className="max-w-md mx-auto">
            <Card className="relative ring-2 ring-primary shadow-xl bg-gradient-to-br from-card to-secondary/30">
              <Badge className="absolute -top-3 left-1/2 transform -translate-x-1/2 bg-gradient-to-r from-primary to-accent text-primary-foreground">
                Premium Plan
              </Badge>
              <CardHeader className="text-center pb-4">
                <CardTitle className="text-2xl">Premium</CardTitle>
                <div className="mt-2">
                  <span className="text-4xl font-bold text-foreground">$9.99</span>
                  <span className="text-muted-foreground">/month</span>
                </div>
              </CardHeader>
              <CardContent className="space-y-6">
                <ul className="space-y-3">
                  {premiumFeatures.map((feature, i) => (
                    <li key={i} className="flex items-center gap-2">
                      <Check className="h-4 w-4 text-success" />
                      <span className="text-sm">{feature}</span>
                    </li>
                  ))}
                </ul>
                <Button 
                  className="w-full bg-gradient-to-r from-primary to-accent hover:from-primary-hover hover:to-accent-hover text-primary-foreground"
                  onClick={handleUpgrade}
                >
                  Upgrade Now
                </Button>
                <p className="text-xs text-muted-foreground text-center">
                  Cancel anytime. No long-term commitments.
                </p>
              </CardContent>
            </Card>
          </div>

          <div className="mt-16 text-center">
            <h2 className="text-2xl font-bold text-foreground mb-8">
              Why Upgrade to Premium?
            </h2>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center space-y-4 p-6 rounded-xl bg-card shadow-lg">
                <div className="w-16 h-16 bg-gradient-to-br from-primary to-primary-hover rounded-full flex items-center justify-center mx-auto">
                  <span className="text-2xl">🎯</span>
                </div>
                <h3 className="text-xl font-semibold">More Accurate</h3>
                <p className="text-muted-foreground">Advanced AI algorithms provide deeper insights into your cat's emotions.</p>
              </div>
              <div className="text-center space-y-4 p-6 rounded-xl bg-card shadow-lg">
                <div className="w-16 h-16 bg-gradient-to-br from-accent to-accent-hover rounded-full flex items-center justify-center mx-auto">
                  <span className="text-2xl">📈</span>
                </div>
                <h3 className="text-xl font-semibold">Track Progress</h3>
                <p className="text-muted-foreground">Complete emotion history and trends to monitor your cat's wellbeing.</p>
              </div>
              <div className="text-center space-y-4 p-6 rounded-xl bg-card shadow-lg">
                <div className="w-16 h-16 bg-gradient-to-br from-success to-success-hover rounded-full flex items-center justify-center mx-auto">
                  <span className="text-2xl">🎥</span>
                </div>
                <h3 className="text-xl font-semibold">Video Analysis</h3>
                <p className="text-muted-foreground">Analyze your cat's body language and behavior through video uploads.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Billing;